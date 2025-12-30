[1] Create agent controller+registery which can be used with any framework
[2] Compatible with a2a.
[3] Use event disptach bus mechanisms to implement independent agent without externally hosting.
[4] flow:
    eg. available agents : X provides a for this needed b, Y provides b needed c, Z provides c, S. where S is supervisor.
    HUMAN -a?-> s, accessible agents s: x, y, z
    S asks X for a: S->X and X requests for b, X-b?->S. accessible agents for S: Y,Z
    S asks Y for b: S->Y and Y requests for c, Y-c?->S. accessible agents for S: Z
    S asks Z for c: S->Z and Z returns c.
    resolves this backwards, Z-c->S-c->Y-b->S-b->X-a->S-a->HUMAN.