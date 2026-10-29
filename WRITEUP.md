# Document Assistant: Write-up

This document explains the design decisions behind my LangGraph document assistant, how state and memory work, how structured output is enforced, and shows example conversations for every feature.

---

## 1. Architecture & Routing Decisions

### Graph structure

```
                      ┌──► qa_agent ────────────┐
                      │                         │
classify_intent ──────┼──► summarization_agent ─┼──► update_memory ──► END
                      │                         │
                      └──► calculation_agent ───┘
```

The graph has five nodes. `classify_intent` is the entry point. It sends each request to exactly one of three specialist agents. Every agent then passes through `update_memory`, which is the last node before `END`.

### Why a classifier plus specialist agents

I separated the work by task type instead of using one general agent. Each task needs different instructions and a different output shape:

- Q&A needs a cited answer.
- Summarization needs a summary plus key points.
- Calculation needs an expression, a result, and an explanation, and must never do arithmetic by itself.

Separate system prompts and schemas keep each agent focused. The routing step is cheap because it is a single structured LLM call.

### How routing works

1. `classify_intent` calls the LLM with the user's input and the conversation history, and gets back a `UserIntent` object.
2. It maps `intent_type` to a `next_step` value:

   | `intent_type` | `next_step` |
   |---|---|
   | `qa` | `qa_agent` |
   | `summarization` | `summarization_agent` |
   | `calculation` | `calculation_agent` |
   | anything else, including `unknown` | `qa_agent` |

3. A conditional edge uses the `should_continue` function, which reads `state["next_step"]`, and a mapping that sends execution to the matching node.
4. Each agent sets `next_step` to `"update_memory"`, and a fixed edge connects each agent to `update_memory`.

I put the routing decision in state (`next_step`) instead of calling the LLM again inside the router. The classifier is the only node that decides, and the router stays a one-line function that is easy to test.

### Decisions worth explaining

- **`unknown` falls back to the Q&A agent.** A greeting or off-topic message still gets a helpful reply, and the user never hits a dead end. The trade-off is that the Q&A agent sometimes searches documents it does not need to.
- **Arithmetic questions about documents are classified as `calculation`, not `qa`.** The classifier prompt states this rule explicitly (for example, "total of all invoices" needs arithmetic). This keeps all math inside the calculator tool, so every number can be traced in the tool log.
- **The classifier prompt contains examples and a confidence scale.** There are three examples per category, rules for ambiguous cases, and scoring bands for `confidence`. The examples calibrate the model, and the scale makes the `confidence` and `reasoning` fields meaningful.
- **One shared `invoke_react_agent` helper.** All three agents call it with their own schema, so they differ only in prompt and schema. Each returns the same state update: `messages`, `actions_taken`, `current_response`, `tools_used`, and `next_step`.
- **The calculation agent is forced through tools.** Its prompt tells it to read the relevant document with `document_reader`, work out the expression, and use the `calculator` tool for every calculation, however simple.
- **The calculator is restricted.** It strips `$` and `,`, validates the expression with a regex that allows only digits, whitespace, `+ - * / % ( ) .`, rejects `**` (to prevent a huge exponent from hanging the process), and runs `eval()` with builtins disabled. Errors return a readable string instead of raising.
- **`update_memory` always runs last.** Summarizing and tracking documents is the same for all intents, so it is one shared node and not repeated in each agent.
- **Tool logs are per session.** `start_session` creates a `ToolLogger` with the session id, so each session gets its own file at `logs/session_<id>.json`. Tools are rebuilt with that logger, and `process_message` passes them to the graph through the config.

---

## 2. State & Memory

### The state object

`AgentState` is a `TypedDict`. Each key is a channel. A **reducer** controls what happens when a node writes to that channel. Channels without a reducer are overwritten.

| Field | Reducer | Purpose |
|---|---|---|
| `messages` | `add_messages` | Full conversation, including tool calls and results. New messages are appended. |
| `actions_taken` | `operator.add` | Names of the nodes that ran. Lists are concatenated, not replaced. |
| `user_input` | overwrite | The current user message |
| `intent` | overwrite | The `UserIntent` object from the classifier |
| `next_step` | overwrite | Routing signal |
| `conversation_summary` | overwrite | Latest summary from `update_memory` |
| `active_documents` | overwrite | Document ids relevant to the latest message |
| `current_response` | overwrite | Raw result of the last agent run, including its structured response |
| `tools_used` | overwrite | Tools called during this turn |
| `session_id`, `user_id` | overwrite | Session bookkeeping |

Without the `operator.add` reducer on `actions_taken`, each node would replace the list and only the last node's name would remain.

### Short-term memory: the checkpointer

The workflow is compiled with `InMemorySaver`. After each node runs, LangGraph saves a checkpoint keyed by `thread_id`. In `process_message` I set `thread_id` to the session id, so every turn in a session reads and writes the same thread.

What happens on each turn:

1. `process_message` builds an initial state with an empty `messages` list, the new `user_input`, and an empty `actions_taken`.
2. LangGraph loads the thread's latest checkpoint and applies the input through the reducers. The empty `messages` list leaves the stored history unchanged, so earlier turns remain available.
3. `classify_intent` sees the stored history, so a follow-up like "add those up" can be understood.
4. The chosen agent receives the history through a `chat_history` placeholder in its prompt, and the current input as the final message.
5. `update_memory` summarizes the history and records the relevant document ids in `conversation_summary` and `active_documents`.
6. The checkpoint is saved for the next turn.

### Three layers of persistence

| Layer | Where | Lifetime | Contents |
|---|---|---|---|
| Graph state | `InMemorySaver` | The running process | Full messages, summary, active documents, action trace |
| Session file | `sessions/<id>.json` | On disk | One record per turn and the accumulated document context |
| Tool log | `logs/session_<id>.json` | On disk | Every tool call with input and output |

I store a plain dictionary per turn (user input, response, intent, tools used, actions, timestamp) in the session file. The raw graph state holds LangChain message objects, which do not serialize to JSON.

### Behavior to be aware of

- **`actions_taken` accumulates across turns.** The reducer appends, and the checkpoint keeps earlier values, so the list grows for the whole thread. The last entries are always the current turn.
- **Resuming from disk restores the session file, not the graph state.** After a process restart, `InMemorySaver` is empty. A resumed session still has its stored history and document context, but the model no longer sees earlier messages. A database-backed checkpointer would remove this limit.
- **History grows.** A long session would need trimming or a rolling summary.

---

## 3. Structured Output

Everywhere the program depends on the shape of an LLM result, I use a Pydantic schema instead of parsing text.

### Intent classification

`llm.with_structured_output(UserIntent)` returns a `UserIntent` instance:

```python
class UserIntent(BaseModel):
    intent_type: Literal["qa", "summarization", "calculation", "unknown"]
    confidence: float   # constrained to 0.0 - 1.0
    reasoning: str
```

`intent_type` is a `Literal`, so the model cannot return a label that the router does not handle. `confidence` is bounded with `ge=0.0, le=1.0`. A result outside these constraints fails validation instead of reaching the routing code.

### Memory update

`llm.with_structured_output(UpdateMemoryResponse)` returns a `summary` string and a list of `document_ids`. These go straight into `conversation_summary` and `active_documents`.

### Agent answers

`create_react_agent(..., response_format=<Schema>)` runs the tool loop, then makes one extra LLM call that formats the final answer into the schema. The validated object is stored under `structured_response` in the agent result, which `current_response` keeps.

| Agent | Schema | Key fields |
|---|---|---|
| `qa_agent` | `AnswerResponse` | `question`, `answer`, `sources`, `confidence`, `timestamp` |
| `summarization_agent` | `SummarizationResponse` | `summary`, `key_points`, `document_ids` |
| `calculation_agent` | `CalculationResponse` | `expression`, `result`, `explanation`, `units` |

### Tool inputs

The `@tool` decorator builds an argument schema from each function's type hints. For example, `document_search` restricts `search_type` and `comparison` to fixed `Literal` values, so the model can only call it with valid arguments.

### Why this matters

- Routing logic never parses free text. It compares a validated string.
- Field types and ranges are checked before any downstream code runs.
- The schemas document what each node must produce.

---

## 4. Example Conversations

Example A is a real session, captured from `sessions/e68bdb3a-933b-47bf-80cd-4ddcebc2b7ca.json`. Long responses are abbreviated; the content is unchanged.

Sample data in `retrieval.py`: `INV-001` (Acme, subtotal $20,000, tax $2,000), `INV-002` (TechStart, $69,300 due), `INV-003` (Global Corp, $214,500 due), `CON-001` (service agreement, $15,000/month, $180,000 total), `CLM-001` (medical claim, $2,450).

### Example A: a four-turn session (summarization, calculation, memory, fallback)

**Turn 1: summarization**

```
User: Summarize all contracts

intent:        summarization (confidence 0.95)
tools_used:    ['document_search', 'document_reader']
actions_taken: ['classify_intent', 'summarization_agent', 'update_memory']

Assistant: ### Contract Summary
Document ID: CON-001 - Service Agreement, January 1, 2024
Parties: DocDacity Solutions Inc. (provider), Healthcare Partners LLC (client)
Services: Document Processing Platform Access, 24/7 Technical Support,
Monthly Data Analytics Reports, Compliance Monitoring
Monthly fee: $15,000. Total contract value: $180,000. Duration: 12 months.
Termination: either party may terminate with 60 days written notice. ...
```

The classifier's `reasoning` field cites the word "summarize" as the deciding signal, and the request is routed to `summarization_agent`. The agent searched for contracts, then read the one it found.

**Turn 2: counting question classified as `calculation`**

```
User: how many contracts with more than 50000 ticket size

intent:        calculation (confidence 0.95)
tools_used:    ['document_search', 'document_reader']
actions_taken: ['classify_intent', 'summarization_agent', 'update_memory',
                'classify_intent', 'calculation_agent', 'update_memory']

Assistant: There is 1 contract with a ticket size of $180,000, which is more
than $50,000.
```

The classifier treats counting items against a threshold as a calculation, as the prompt rule for document questions that need computation says. The calculation agent answered from document data. A count of one match needs no arithmetic, so it did not call the `calculator` tool. `actions_taken` now holds both turns' nodes because of the `operator.add` reducer and checkpointer.

**Turn 3: a reference resolved from memory ("this contract")**

```
User: give the details of this contract

intent:        summarization (confidence 0.90)
tools_used:    ['document_search', 'document_reader']
actions_taken: [... previous six entries ..., 'classify_intent',
                'summarization_agent', 'update_memory']

Assistant: ### Contract Details
Document ID: CON-001 - Service Agreement ... (same details as turn 1)
```

"This contract" has no meaning on its own. The classifier's reasoning notes that the conversation history shows the user focused on contracts, and the agent returns the details of `CON-001`. This shows that history carries across turns through the checkpointer.

**Turn 4: unclear input, fallback routing**

```
User: end

intent:        unknown (confidence 0.50)  -> routed to qa_agent by default
actions_taken: 12 entries (3 nodes x 4 turns), ending with
               'classify_intent', 'qa_agent', 'update_memory'

Assistant: If you have any more questions in the future, feel free to ask.
Have a great day!
```

The classifier gives a low confidence (0.50) and `unknown`, as the confidence scale in the prompt instructs for ambiguous input. The router sends `unknown` to the Q&A agent, so the user still gets a sensible reply.

**Resulting session file**

After these four turns, the session file holds one JSON record per turn and the accumulated document context:

```json
{
  "session_id": "e68bdb3a-933b-47bf-80cd-4ddcebc2b7ca",
  "user_id": "demo_user",
  "conversation_history": [
    {
      "user_input": "give the details of this contract",
      "response": "### Contract Details ...",
      "intent": {"intent_type": "summarization", "confidence": 0.9, "reasoning": "..."},
      "tools_used": ["document_search", "document_reader"],
      "actions_taken": ["classify_intent", "summarization_agent", "update_memory", "..."],
      "timestamp": "2026-10-29T07:52:45.833043"
    }
  ],
  "document_context": ["CON-001"],
  "created_at": "2026-10-29T07:50:17.225687",
  "last_updated": "2026-10-29T07:53:21.864935"
}
```

### Example B: calculation with the calculator tool (to be captured)

Example A did not use the calculator, so this example is still needed. Run a question that needs real arithmetic, for example:

```
User: What is the total of INV-002 and INV-003?
```

Expected behavior: intent `calculation`, `tools_used` includes `document_reader` and `calculator`, and the answer is $283,800 (69,300 + 214,500). Paste the real output here:

```
<paste intent, tools_used, sources, actions_taken and response>
```

Then paste the matching entry from `logs/session_<id>.json`. Each entry has this shape:

```json
{
  "timestamp": "...",
  "tool_name": "calculator",
  "input": {"expression": "..."},
  "output": "{'result': ...}"
}
```

### Example C: resuming a session (to be captured)

```python
assistant.start_session(user_id="demo_user", session_id="e68bdb3a-933b-47bf-80cd-4ddcebc2b7ca")
# prints: Resumed session e68bdb3a-933b-47bf-80cd-4ddcebc2b7ca
```

A resumed session restores its stored turn history and document context (`CON-001`) from `sessions/<id>.json`. Within one process the in-memory checkpoint also remains, so the model still remembers earlier messages. After a restart, only the session file is restored. Paste the output of a resumed follow-up message here.

### Script to reproduce

```python
import os
from assistant import DocumentAssistant

assistant = DocumentAssistant(openai_api_key=os.getenv("OPENAI_API_KEY"))
sid = assistant.start_session(user_id="demo_user")

messages = [
    "Summarize all contracts",
    "how many contracts with more than 50000 ticket size",
    "give the details of this contract",
    "What invoices do we have over $50,000?",
    "What is the total of INV-002 and INV-003?",
    "end",
]

for msg in messages:
    result = assistant.process_message(msg)
    print(f"\nUser: {msg}")
    print("success:      ", result["success"], result.get("error", ""))
    print("intent:       ", result.get("intent"))
    print("tools_used:   ", result.get("tools_used"))
    print("sources:      ", result.get("sources"))
    print("actions_taken:", result.get("actions_taken"))
    print("Assistant:    ", result.get("response"))

assistant.start_session(user_id="demo_user", session_id=sid)   # resume
```