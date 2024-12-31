#ifndef _queue_h_INCLUDED
#define _queue_h_INCLUDED

#define DISCONNECT UINT_MAX
#define DISCONNECTED(IDX) ((int)(IDX) < 0)

struct kissat;

typedef struct links links;
typedef struct queue queue;

struct links
{
  unsigned prev, next;
  unsigned stamp;
};

struct queue
{
  unsigned first, last, stamp;
  struct
  {
    unsigned idx, stamp;
  } search;
};

void
kissat_init_queue (queue* queue);
void
kissat_reset_search_of_queue (links* links, queue* queue);

void
kissat_reassign_queue_stamps (kissat * solver, links* links, queue* queue);

#define LINK_FOCUSED(IDX) \
  (solver->links_focused[assert ((IDX) < VARS), (IDX)])

#define LINK_UNSATORD(IDX) \
  (solver->links_unsatord[assert ((IDX) < VARS), (IDX)])

#if defined(CHECK_QUEUE) && !defined(NDEBUG)
void
kissat_check_queue (kissat * solver, links* links, queue* queue);
#else
#define kissat_check_queue(...) do { } while (0)
#endif


#endif
