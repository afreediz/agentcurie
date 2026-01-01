U -urgent, O -ongoing, F -future

[U] Fix agent & tool model for better prediction schema and proper logic.
[U] Enable multi tool calling but single agent calling.
[U] Move to asyncio.Event/Future to avoid unnecessary task wakeup and latency.

[F] Improve messages manager, efficently manages history using lesser tokens, summarizing and caching.
[F] Make agent registeries to dynamically fetch params (with types) needed and guide the model.
[F] Update mcp_client to be able to communicate with remote servers.
[F] Develop mcp similar protocol to enable remote agent communication.
[F] Parallel agent executions and coordination where supervisor act as judge.