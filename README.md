# Document Assistant

A multi-agent document assistant built with LangChain and LangGraph. It answers questions about financial and healthcare documents, summarizes them, and performs calculations on their data. It remembers context across turns within a session.

## Contents

1. [Quick start](#quick-start)
2. [Architecture](#architecture)
3. [Implementation decisions](#implementation-decisions)
4. [How state and memory work](#how-state-and-memory-work)
5. [How structured outputs are enforced](#how-structured-outputs-are-enforced)
6. [Example conversations](#example-conversations)
7. [Logs and session files](#logs-and-session-files)
8. [Known limitations](#known-limitations)

---

## Quick start

```bash
cp .env.example .env        # add your OpenAI API key
pip install -r requirements.txt
python main.py
```

Project layout:

```
doc_assistant_project/
├── src/
│   ├── schemas.py        # Pydantic models (structured outputs, session state)
│   ├── retrieval.py      # Simulated document retriever (in-memory sample docs)
│   ├── tools.py          # Agent tools + ToolLogger
│   ├── prompts.py        # Prompt templates
│   ├── agent.py          # LangGraph workflow (nodes, state, routing)
│   └── assistant.py      # DocumentAssistant: sessions + workflow invocation
├── sessions/             # Saved conversation sessions (one JSON per session)
├── logs/                 # Tool call logs (one JSON per session)
├── main.py
└── requirements.txt
```

---

## Architecture

```
                      ┌──► qa_agent ────────────┐
                      │                         │
classify_intent ──────┼──► summarization_agent ─┼──► update_memory ──► END
                      │                         │
                      └──► calculation_agent ───┘
```

| Node | Responsibility |
|---|---|
| `classify_intent` | Asks the LLM to label the request as `qa`, `summarization`, `calculation`, or `unknown`, and sets `next_step`. |
| `qa_agent` | ReAct agent that searches and reads documents, then answers with sources. |
| `summarization_agent` | ReAct agent that reads documents and produces a summary and key points. |
| `calculation_agent` | ReAct agent that reads documents and uses the calculator tool for all math. |
| `update_memory` | Summarizes the conversation and records which documents are relevant. |

The three agents share one implementation pattern (`invoke_react_agent`) and differ only in their system prompt and output schema.

---

## Implementation decisions

### Schemas (`schemas.py`)

- **`UserIntent.intent_type` is a `Literal`.** The four allowed values are enforced by the type, so the model cannot return a label the router does not know. `confidence` is bounded with `ge=0.0, le=1.0`.
- **`AnswerResponse`** carries `question`, `answer`, `sources`, `confidence`, and a `timestamp` that defaults to the current time.
- **`UpdateMemoryResponse`** has two fields: `summary` (string) and `document_ids` (list of strings). These map directly onto the `conversation_summary` and `active_documents` state fields.
- **`SessionState.conversation_history`** is a list of plain dictionaries (user input, response, intent, tools used, actions, timestamp). Plain dictionaries are used so the whole session can be written with `json.dump`.

### Intent classification (`classify_intent`)

- The LLM is wrapped with `llm.with_structured_output(UserIntent)`, so the result is a validated `UserIntent` object rather than text to parse.
- The prompt comes from `get_intent_classification_prompt()` and is filled with `user_input` and `conversation_history`. It defines each category with three examples, rules for ambiguous cases (for example, any arithmetic means `calculation`), and a confidence scale with instructions for the `reasoning` field.
- Routing is a plain `if/elif` on `intent_type`:

  | `intent_type` | `next_step` |
  |---|---|
  | `qa` | `qa_agent` |
  | `summarization` | `summarization_agent` |
  | `calculation` | `calculation_agent` |
  | anything else (including `unknown`) | `qa_agent` |

- `unknown` falls back to the Q&A agent so the user always gets a response. A greeting or off-topic question is answered by the Q&A agent instead of hitting a dead end.
- The node stores the whole `UserIntent` object in `state["intent"]`, matching the `Optional[UserIntent]` type in `AgentState`. The confidence and reasoning stay available for logging and debugging.

### Agents (`qa_agent`, `summarization_agent`, `calculation_agent`)

- Each agent looks up its prompt with `get_chat_prompt_template(<intent>)`, which returns a system prompt, the chat history placeholder, and the user input.
- `invoke_react_agent` builds a LangGraph ReAct agent with the tools and the agent's response schema, runs it, and collects the names of the tools that were called.
- Each agent returns the same state update: `messages`, `actions_taken`, `current_response`, `tools_used`, and `next_step` (set to `update_memory`).
- The calculation prompt instructs the model to identify and read the needed document with `document_reader`, work out the expression, and use the `calculator` tool for every calculation, however simple. This keeps arithmetic out of the model's head and makes every number traceable in the tool log.

### Memory update (`update_memory`)

- Reads the LLM from `config["configurable"]["llm"]`, which is why the node takes a `config` argument.
- Runs `MEMORY_SUMMARY_PROMPT` over the full message history and asks for an `UpdateMemoryResponse`.
- Writes `conversation_summary`, `active_documents`, and sets `next_step` to `"end"`. It also appends `"update_memory"` to `actions_taken` so the action trace covers every node.

### Workflow (`create_workflow`)

- All five nodes are registered. `classify_intent` is the entry point.
- A conditional edge reads `state["next_step"]` through `should_continue` and maps it to one of the three agent nodes. The `"end"` mapping is a safety net and is not used by `classify_intent`, which always picks an agent.
- Each agent has a fixed edge to `update_memory`, and `update_memory` goes to `END`.
- The graph is compiled with `InMemorySaver()` so state persists per `thread_id`.

### Tools (`tools.py`)

- **Calculator.** The expression is cleaned (`$` and `,` removed), then validated with a regex that allows only digits, whitespace, `+ - * / % ( ) .`. The `**` operator is rejected explicitly, because an expression like `9**9**9` would run for a very long time. Evaluation uses `eval()` with `{"__builtins__": {}}` and no locals. Division by zero and any other error return a readable error string instead of raising. Every call, successful or not, is logged.
- **Document search, reader, and statistics** wrap the retriever and log each call.
- All tools return strings and log through the shared `ToolLogger`.

### Prompts (`prompts.py`)

- `get_chat_prompt_template` picks the system prompt for `qa`, `summarization`, or `calculation`, and falls back to the Q&A prompt for anything else.
- System prompts contain no curly braces, because `SystemMessagePromptTemplate` treats `{text}` as a template variable.

### Sessions and logs (`assistant.py`)

- `start_session` creates a new `SessionState` or resumes one from `sessions/<session_id>.json`, then creates a `ToolLogger` with that session id. Tool calls are therefore written to `logs/session_<session_id>.json`, one file per session.
- `process_message` builds the config the graph needs (`thread_id`, `llm`, `tools`), invokes the workflow, appends a JSON-friendly record of the turn to the session history, and saves the session.
- The raw graph state is not stored in the session file. It contains LangChain message objects, which do not serialize to JSON.

---

## How state and memory work

### The state object

`AgentState` is a `TypedDict`. Each key is a channel. A **reducer** decides what happens when a node writes to a channel; channels without a reducer are simply overwritten.

| Field | Reducer | Purpose |
|---|---|---|
| `messages` | `add_messages` | Full conversation, including tool calls and results. New messages are appended, and messages with the same id replace each other. |
| `actions_taken` | `operator.add` | Names of the nodes that ran. Lists are concatenated instead of overwritten. |
| `user_input` | overwrite | The current user message. |
| `intent` | overwrite | The `UserIntent` from the classifier. |
| `next_step` | overwrite | Routing signal read by `should_continue`. |
| `conversation_summary` | overwrite | Latest summary written by `update_memory`. |
| `active_documents` | overwrite | Document ids relevant to the latest message. |
| `current_response` | overwrite | Raw result of the last agent run, including its structured response. |
| `tools_used` | overwrite | Tool names called during the current turn. |
| `session_id`, `user_id` | overwrite | Session bookkeeping. |

### The checkpointer and `thread_id`

The workflow is compiled with `InMemorySaver`. After each node, LangGraph saves a checkpoint keyed by `config["configurable"]["thread_id"]`. `process_message` sets that to the session id, so every turn in a session reads and writes the same thread.

What happens on each turn:

1. `process_message` builds `initial_state` with `messages: []`, the new `user_input`, and an empty `actions_taken`.
2. On invoke, LangGraph loads the thread's latest checkpoint and applies the input through the reducers. The empty `messages` list leaves the stored history unchanged, so earlier turns are still present. Fields without reducers take the input values.
3. `classify_intent` sees the stored message history, so follow-ups such as "add those up" can be resolved.
4. The chosen agent receives the history through the `chat_history` placeholder, and the current input as the final human message.
5. `update_memory` summarizes the full history and records the relevant document ids.
6. The checkpoint is saved, ready for the next turn.

### Three layers of persistence

| Layer | Where | Lifetime | Contents |
|---|---|---|---|
| Graph state | `InMemorySaver` | The running process | Full message history, summary, active documents, action trace |
| Session file | `sessions/<id>.json` | On disk | One record per turn, plus the accumulated document context |
| Tool log | `logs/session_<id>.json` | On disk | Every tool call with its input and output |

### Things to know

- **`actions_taken` accumulates across turns.** The reducer appends and the checkpoint keeps the previous value, so the list grows for the life of the thread. The last three entries are always the nodes of the current turn.
- **Resuming from disk restores the session file, not the graph state.** If the process restarts, `InMemorySaver` is empty. The resumed session still has its stored history and document context, but the model no longer sees earlier messages. Replacing `InMemorySaver` with a database-backed checkpointer would remove this limitation.

---

## How structured outputs are enforced

Every place where the program depends on the shape of an LLM result uses a Pydantic schema instead of parsing text.

1. **Intent classification.** `llm.with_structured_output(UserIntent)` returns a `UserIntent` instance. `intent_type` is a `Literal` of four values and `confidence` must be between 0 and 1. A response outside those constraints fails validation instead of reaching the router.
2. **Memory update.** `llm.with_structured_output(UpdateMemoryResponse)` returns a `summary` string and a list of `document_ids`. These values are written straight into `conversation_summary` and `active_documents`.
3. **Agent answers.** `create_react_agent(..., response_format=<Schema>)` runs the tool loop, then makes one extra LLM call that formats the final answer into the schema. The validated object is stored under `structured_response` in the agent result, which `current_response` keeps. The mapping is:

   | Agent | Schema |
   |---|---|
   | `qa_agent` | `AnswerResponse` |
   | `summarization_agent` | `SummarizationResponse` |
   | `calculation_agent` | `CalculationResponse` |

4. **Tool inputs.** The `@tool` decorator builds an argument schema from each function's type hints. For example, `document_search` restricts `search_type` and `comparison` to fixed `Literal` values, so the model can only call it with valid arguments.

---

## Example conversations

> These transcripts show the expected behavior for the sample documents in `retrieval.py`. Exact wording from the model will vary. Replace them with output from your own run before submitting.

Sample data used below: `INV-001` (Acme, subtotal $20,000, tax $2,000), `INV-002` (TechStart, $69,300 due), `INV-003` (Global Corp, $214,500 due), `CON-001` (service agreement, $15,000/month, $180,000 total), `CLM-001` (medical claim, $2,450).

### Example A: a four-turn session (Q&A, calculation, summarization, fallback)

**Turn 1: Q&A**

```
User: What invoices do we have over $50,000?

intent:        qa (confidence 0.95)
tools_used:    ['document_search']
sources:       ['INV-002', 'INV-003']
actions_taken: ['classify_intent', 'qa_agent', 'update_memory']

Assistant: Two invoices are over $50,000: INV-002 (TechStart Inc., $69,300
total due) and INV-003 (Global Corp, $214,500 total due).
```

**Turn 2: calculation, with the follow-up resolved from history**

```
User: Add those two up.

intent:        calculation (confidence 0.93)
tools_used:    ['calculator']
sources:       ['INV-002', 'INV-003']
actions_taken: ['classify_intent', 'qa_agent', 'update_memory',
                'classify_intent', 'calculation_agent', 'update_memory']

Assistant: The combined total due is $283,800.
Expression: 69300 + 214500 (INV-002 and INV-003).
```

The classifier resolves "those two" from the message history. `actions_taken` shows the previous turn's entries followed by this turn's, because of the reducer and checkpointer.

**Turn 3: summarization of a reference ("the larger one")**

```
User: Summarize the larger one.

intent:        summarization (confidence 0.88)
tools_used:    ['document_reader']
sources:       ['INV-003']

Assistant: INV-003 is an invoice dated 2024-03-01 for Global Corp. It covers an
annual subscription ($120,000), premium support ($30,000) and custom development
($45,000), for a subtotal of $195,000. With 10% tax ($19,500), the total due is
$214,500, on Net 60 payment terms.
```

**Turn 4: out-of-scope input, routed to the fallback**

```
User: Hello

intent:        unknown (confidence 0.90)  -> routed to qa_agent by default
tools_used:    []

Assistant: Hello! I can answer questions about your documents, summarize them,
or run calculations on their figures.
```

### Example B: calculation that needs a document and the calculator

```
User: What is the monthly fee on CON-001 multiplied by 12?

intent:        calculation (confidence 0.92)
tools_used:    ['document_reader', 'calculator']
sources:       ['CON-001']

Assistant: The monthly fee in CON-001 is $15,000. Over 12 months,
15000 * 12 = $180,000, which matches the stated total contract value.
```

The calculation prompt requires the document to be read first and the calculator to perform the multiplication. The tool log records both calls.

### Example C: resuming a session

```python
assistant = DocumentAssistant(openai_api_key=key)
sid = assistant.start_session(user_id="demo_user")
assistant.process_message("What invoices do we have over $50,000?")

# later, possibly in a new process
assistant.start_session(user_id="demo_user", session_id=sid)   # "Resumed session ..."
```

The resumed session restores its stored turn history and document context from `sessions/<sid>.json`. Within the same process, the in-memory checkpoint is still available, so the model also remembers the earlier messages. After a restart only the session file is restored (see [Things to know](#things-to-know)).

---

## Logs and session files

Session file (`sessions/<session_id>.json`), abbreviated:

```json
{
  "session_id": "…",
  "user_id": "demo_user",
  "conversation_history": [
    {
      "user_input": "Add those two up.",
      "response": "The combined total due is $283,800. …",
      "intent": {"intent_type": "calculation", "confidence": 0.93, "reasoning": "…"},
      "tools_used": ["calculator"],
      "actions_taken": ["classify_intent", "…"],
      "timestamp": "2024-…"
    }
  ],
  "document_context": ["INV-002", "INV-003"],
  "created_at": "…",
  "last_updated": "…"
}
```

Tool log (`logs/session_<session_id>.json`), abbreviated:

```json
[
  {
    "timestamp": "…",
    "tool_name": "calculator",
    "input": {"expression": "69300 + 214500"},
    "output": "{'result': 283800}"
  }
]
```

To produce these files, run a short script that sends a Q&A, a calculation, and a follow-up message (as in Example A). The `logs/` directory is created when a `ToolLogger` is constructed, and the `sessions/` directory when `DocumentAssistant` starts. The files themselves appear after the first successful message.

---

## Known limitations

- **Process-local graph memory.** `InMemorySaver` is lost on restart. See [Things to know](#things-to-know).
- **Growing history.** Each agent run can re-add its system prompt to `messages`, so history grows by a few messages per turn. A long session would need trimming or a summary-based window.
- **Calculator scope.** It supports `+ - * / % ( )` and decimals only. Exponents and functions are deliberately not allowed.
- **Simulated retrieval.** Documents are five in-memory samples searched by keyword and amount, not a vector store.
- **Fallback routing.** `unknown` intents are answered by the Q&A agent rather than refused or clarified.