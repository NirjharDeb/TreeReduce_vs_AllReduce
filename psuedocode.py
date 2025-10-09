function initiate_global_done_tree():
    mark LOCAL_DONE[me] = -1
    record ELAPSED_MS[me]

    loop forever:
        # top-level detection & coordinated exit
        if shmem_get(GROUP_DONE[top][0], ROOT_PE) == 1:
            if me == ROOT_PE:
                if atomic_CAS(AGG_PRINTED@ROOT, 0 -> 1) == 0:
                    collect all ELAPSED_MS (local + shmem_get from others)
                    compute min, avg, max
                    print aggregate
                put ROOT_GO@ROOT = 1
                wait_until(EXIT_ACKS@ROOT >= npes - 1)
                shmem_global_exit(0)
            else:
                while shmem_get(ROOT_GO, ROOT_PE) == 0:
                    tiny_pause()
                atomic_fetch_inc(EXIT_ACKS@ROOT)
                shmem_global_exit(0)

        # leaf-phase attempt: elect dynamic leader if last finisher in leaf
        try_mark_leaf_group_done(me)

        tiny_pause()


function try_mark_leaf_group_done(me):
    L = 0
    gidx = me / group_span_at_level(G_LEAF, 0)
    host = static_group_owner_pe(G_LEAF, 0, gidx)

    if shmem_get(GROUP_DONE[0][gidx], host) == 1:
        return

    gsize = actual_size_of_leaf_group(gidx)  # handles tail group
    prior = atomic_fetch_inc(LEAF_COUNT[gidx] @ host)

    if prior == gsize - 1:
        complete_group_and_maybe_propagate(me, L=0, gidx=gidx)


function complete_group_and_maybe_propagate(me, L, gidx):
    host = static_group_owner_pe(G_LEAF, L, gidx)
    atomic_CAS(GROUP_DONE[L][gidx] @ host, 0 -> 1)
    put GROUP_LEADER[L][gidx] @ host = me

    while L + 1 < MAX_LEVELS:
        parent_L   = L + 1
        parent_idx = gidx / 2
        phost = static_group_owner_pe(G_LEAF, parent_L, parent_idx)

        expected_children = 1 + (right_child_idx(parent_idx) < NUM_GROUPS[L] ? 1 : 0)
        prior = atomic_fetch_inc(CHILD_DONE_COUNT[parent_L][parent_idx] @ phost)

        if prior + 1 == expected_children:
            atomic_CAS(GROUP_DONE[parent_L][parent_idx] @ phost, 0 -> 1)
            put GROUP_LEADER[parent_L][parent_idx] @ phost = me
            L    = parent_L
            gidx = parent_idx
        else:
            break
