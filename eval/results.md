# W16 agent evaluation

Agent model: `groq/openai/gpt-oss-20b` - generated 2026-09-19 21:17

## Summary

| context_mode | n | completion_rate | tools_ok_rate | arg_valid_rate | avg_steps | steps_in_range | avg_tokens | total_tokens |
|---|---|---|---|---|---|---|---|---|
| notes | 12 | 0.75 | 0.83 | 1.0 | 4.5 | 0.83 | 4706 | 56476 |
| full | 12 | 0.92 | 1.0 | 1.0 | 4.0 | 1.0 | 4579 | 54950 |

## Per-case results - context mode `notes`

| id | completed | tools_ok | call_valid_rate | steps | steps_ok | tokens | confidence | tools |
|---|---|---|---|---|---|---|---|---|
| single_fact | True | True | 1.0 | 3 | True | 2937 | 0.95 | ['search_documents', 'record_note'] |
| storage | True | True | 1.0 | 3 | True | 2931 | 1.0 | ['search_documents', 'record_note'] |
| zenith_price | True | True | 1.0 | 3 | True | 2917 | 1.0 | ['search_documents', 'record_note'] |
| conflict_latest_price | True | True | 1.0 | 5 | True | 5705 | 0.95 | ['search_documents', 'record_note', 'search_documents', 'record_note'] |
| retrieve_then_calc | True | False | 1.0 | 3 | True | 2930 | 0.95 | ['search_documents', 'record_note'] |
| annual_discount | True | False | 1.0 | 3 | False | 3051 | 1.0 | ['search_documents', 'record_note'] |
| compare_recommend | False | True | 1.0 | 9 | True | 10522 | 0.2 | ['search_documents', 'search_documents', 'calculator', 'search_documents', 'search_documents', 'record_note', 'search_documents', 'record_note'] |
| calc_only | True | True | 1.0 | 4 | True | 3507 | 1.0 | ['calculator', 'calculator', 'record_note'] |
| datetime_only | True | True | 1.0 | 3 | True | 2515 | 1.0 | ['current_datetime', 'record_note'] |
| not_in_docs | False | True | 1.0 | 9 | False | 10669 | 0.2 | ['search_documents', 'search_documents', 'search_documents', 'search_documents', 'search_documents', 'search_documents', 'search_documents', 'search_documents'] |
| ambiguous_asks | False | True | 1.0 | 3 | True | 2863 | 0.95 | ['search_documents', 'record_note'] |
| multi_turn | True | True | 1.0 | 6 | True | 5929 | 1.0 | ['search_documents', 'record_note', 'search_documents', 'record_note'] |

### Failure log (`notes`)

| id | class | first_error | tools | confidence | answer |
|---|---|---|---|---|---|
| retrieve_then_calc | soft | None | ['search_documents', 'record_note'] | 0.95 | $280 per month |
| annual_discount | soft | None | ['search_documents', 'record_note'] | 1.0 | The yearly cost for 4 Acme Pro seats with annual billing is $1,344. |
| compare_recommend | hard | None | ['search_documents', 'search_documents', 'calculator', 'search_documents', 'search_documents', 'record_note', 'search_documents', 'record_note'] | 0.2 | I could not finish within 8 steps. What I verified so far: Acme Starter costs $12 per seat per month |
| not_in_docs | hard | None | ['search_documents', 'search_documents', 'search_documents', 'search_documents', 'search_documents', 'search_documents', 'search_documents', 'search_documents'] | 0.2 | I could not finish within 8 steps. |
| ambiguous_asks | soft | None | ['search_documents', 'record_note'] | 0.95 | The Starter plan costs $12 per seat per month. |

### Failure injection (`notes`)

Pass = the agent recognized the failure and did not give a confident answer.

| id | fault | failure_recognized | confidence | pass | steps | tokens | answer |
|---|---|---|---|---|---|---|---|
| single_fact | search_down | True | 0.2 | True | 4 | 3577 | [Some sources were unavailable or returned invalid data; this is unverified.] I’m sorry, but I couldn’t find any information in the provided documents about the |
| retrieve_then_calc | search_down | True | 0.3 | True | 4 | 3543 | [Some sources were unavailable or returned invalid data; this is unverified.] I’m sorry, but I couldn’t find any information in the provided documents about the |
| compare_recommend | search_down | True | 0.0 | True | 6 | 5839 | [Some sources were unavailable or returned invalid data; this is unverified.] I’m sorry, but the uploaded documents do not contain the pricing information neede |
| single_fact | malformed | True | 0.2 | True | 5 | 4489 | [Some sources were unavailable or returned invalid data; this is unverified.] I could not find a verified source for the Starter plan price per seat per month a |
| retrieve_then_calc | malformed | True | 0.2 | True | 4 | 3490 | [Some sources were unavailable or returned invalid data; this is unverified.] I could not find any information in the provided documents that specifies the cost |
| compare_recommend | malformed | True | 0.0 | True | 5 | 4655 | [Some sources were unavailable or returned invalid data; this is unverified.] I’m sorry, but I couldn’t find any pricing information for Acme Starter, Acme Pro, |

## Per-case results - context mode `full`

| id | completed | tools_ok | call_valid_rate | steps | steps_ok | tokens | confidence | tools |
|---|---|---|---|---|---|---|---|---|
| single_fact | True | True | 1.0 | 3 | True | 3205 | 1.0 | ['search_documents', 'record_note'] |
| storage | True | True | 1.0 | 3 | True | 3180 | 1.0 | ['search_documents', 'record_note'] |
| zenith_price | True | True | 1.0 | 3 | True | 3154 | 1.0 | ['search_documents', 'record_note'] |
| conflict_latest_price | True | True | 1.0 | 4 | True | 4528 | 0.95 | ['search_documents', 'current_datetime', 'record_note'] |
| retrieve_then_calc | True | True | 1.0 | 4 | True | 4414 | 0.95 | ['search_documents', 'calculator', 'record_note'] |
| annual_discount | True | True | 1.0 | 4 | True | 4657 | 1.0 | ['search_documents', 'record_note', 'calculator'] |
| compare_recommend | False | True | 1.0 | 8 | True | 11041 | 0.0 | ['search_documents', 'search_documents', 'calculator', 'record_note', 'search_documents', 'record_note', 'calculator'] |
| calc_only | True | True | 1.0 | 4 | True | 3436 | 1.0 | ['calculator', 'calculator', 'record_note'] |
| datetime_only | True | True | 1.0 | 2 | True | 1605 | 1.0 | ['current_datetime'] |
| not_in_docs | True | True | 1.0 | 5 | True | 7483 | 0.3 | ['search_documents', 'search_documents', 'search_documents', 'search_documents'] |
| ambiguous_asks | True | True | 1.0 | 2 | True | 1879 | 0.0 | ['search_documents'] |
| multi_turn | True | True | 1.0 | 6 | True | 6368 | 0.95 | ['search_documents', 'record_note', 'search_documents', 'record_note'] |

### Failure log (`full`)

| id | class | first_error | tools | confidence | answer |
|---|---|---|---|---|---|
| compare_recommend | hard | model call failed: Groq error: Error code: 400 - {'error': {'message': 'Failed t | ['search_documents', 'search_documents', 'calculator', 'record_note', 'search_documents', 'record_note', 'calculator'] | 0.0 | The assistant could not reach the model, so no answer was produced. |

### Failure injection (`full`)

Pass = the agent recognized the failure and did not give a confident answer.

| id | fault | failure_recognized | confidence | pass | steps | tokens | answer |
|---|---|---|---|---|---|---|---|
| single_fact | search_down | True | 0.2 | True | 6 | 5458 | [Some sources were unavailable or returned invalid data; this is unverified.] I’m sorry, but I couldn’t find any information in the documents about the Starter  |
| retrieve_then_calc | search_down | True | 0.2 | True | 5 | 4423 | [Some sources were unavailable or returned invalid data; this is unverified.] I’m sorry, but I couldn’t find any information in the provided documents about the |
| compare_recommend | search_down | True | 0.2 | True | 6 | 5582 | [Some sources were unavailable or returned invalid data; this is unverified.] I’m sorry, but I couldn’t find any information in the provided documents about the |
| single_fact | malformed | True | 0.3 | True | 5 | 4385 | [Some sources were unavailable or returned invalid data; this is unverified.] I’m sorry, but I couldn’t find any information in the provided documents about the |
| retrieve_then_calc | malformed | True | 0.3 | True | 4 | 3413 | [Some sources were unavailable or returned invalid data; this is unverified.] I could not find any evidence in the provided documents to determine the cost of 8 |
| compare_recommend | malformed | True | 0.1 | True | 6 | 5419 | I could not verify an answer: a required tool failed or returned invalid data, and I have no reliable evidence. Please try again later. |

## Context-engineering effect

Average tokens per query: `notes` = 4706, `full` = 4579 (completion 0.75 vs 0.92). Tokens are Gemini prompt+output tokens for the agent loop; embedding calls are not counted.
