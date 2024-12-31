#include "decide.h"
#include "inlineframes.h"
#include "inlineheap.h"
#include "inlinequeue.h"

#include <inttypes.h>

static unsigned
last_enqueued_unassigned_variable (kissat * solver, 
  const links* links, queue* queue)
{
  assert (solver->unassigned);
  //const links *const links = solver->links;
  const value *const values = solver->values;
  unsigned res = queue->search.idx;
  if (values[LIT (res)])
    {
      do
	{
	  res = links[res].prev;
	  assert (!DISCONNECTED (res));
	}
      while (values[LIT (res)]);
      kissat_update_queue (links, queue, res);
    }
#ifdef LOGGING
  const unsigned stamp = links[res].stamp;
  LOG ("last enqueued unassigned %s stamp %u", LOGVAR (res), stamp);
#endif
#ifdef CHECK_QUEUE
  for (unsigned i = links[res].next; !DISCONNECTED (i); i = links[i].next)
    assert (VALUE (LIT (i)));
#endif
  return res;
}

static unsigned
largest_score_unassigned_variable (kissat * solver)
{
  heap *scores = SCORES;
  unsigned res = kissat_max_heap (scores);
  const value *const values = solver->values;
  while (values[LIT (res)])
    {
      kissat_pop_max_heap (solver, scores);
      res = kissat_max_heap (scores);
    }
#if defined(LOGGING) || defined(CHECK_HEAP)
  const double score = kissat_get_heap_score (scores, res);
#endif
  LOG ("largest score unassigned %s score %g", LOGVAR (res), score);
#ifdef CHECK_HEAP
  for (all_variables (idx))
    {
      if (!ACTIVE (idx))
	continue;
      if (VALUE (LIT (idx)))
	continue;
      const double idx_score = kissat_get_heap_score (scores, idx);
      assert (score >= idx_score);
    }
#endif
  return res;
}

unsigned decide_next_unsatord(kissat * solver) {
  node* prev_node = NULL;
  unsigned iidx;
  node* tmp;
  node* current = solver->current_unsatord_node;
  list* list = &(solver->unsatord_list);
  while(current !=NULL) {
    iidx = current->var_idx;
    
    if(VALUE(LIT(iidx))) {
      prev_node = current;
      current = current->next;
      continue;
    }
    
    solver->current_unsatord_node = current;
    //printf("iidx %d\n", iidx);
    return iidx;  
  }
  assert(false);
  //printf("false\n", iidx);
  return 0;
}


unsigned
kissat_next_decision_variable (kissat * solver)
{
  unsigned res;
  if (solver->stable) {
    if(solver->apply_unsatord_stable) {
      res = decide_next_unsatord(solver);
      if (res == 0) {
        largest_score_unassigned_variable (solver);
        solver->apply_unsatord_stable = false;
      }
    }
    else
      res = largest_score_unassigned_variable (solver);
  }
  else {
    if(solver->apply_unsatord_focused) {
      res = decide_next_unsatord(solver);
    }
    else
    res = last_enqueued_unassigned_variable (solver, 
      solver->links, &(solver->queue));
  }
  LOG ("next decision %s", LOGVAR (res));
  return res;
}

static inline value
decide_phase (kissat * solver, unsigned idx)
{
  bool force = GET_OPTION (forcephase);

  value *target;
  if (force)
    target = 0;
  else if (!GET_OPTION (target))
    target = 0;
  else if (solver->stable || GET_OPTION (target) > 1)
    target = solver->phases.target + idx;
  else
    target = 0;

  value *saved;
  if (force)
    saved = 0;
  else if (GET_OPTION (phasesaving))
    saved = solver->phases.saved + idx;
  else
    saved = 0;

  value res = 0;

  if(GET_OPTION (neural_backbone_always)) {
      value * neural = solver->phases.neural;
        if(!res && neural)
           res = *(neural + idx);
  }

  if (!res && target && (res = *target))
    {
      LOG ("%s uses target decision phase %d", LOGVAR (idx), (int) res);
      INC (target_decisions);
    }

  if (!res && saved && (res = *saved))
    {
      LOG ("%s uses saved decision phase %d", LOGVAR (idx), (int) res);
      INC (saved_decisions);
    }

  if (!res)
    {
      if(GET_OPTION (neural_backbone_initial) || GET_OPTION (random_phase_initial)) {
        value *initial = solver->phases.initial + idx;
        res = *initial; 
      }
      else {        
        res = INITIAL_PHASE;
      }
      LOG ("%s uses initial decision phase %d", LOGVAR (idx), (int) res);
      INC (initial_decisions);
    }
  assert (res);

  return res;
}

void
kissat_decide (kissat * solver)
{
  START (decide);
  assert (solver->unassigned);
  INC (decisions);
  if (solver->stable)
    INC (stable_decisions);
  else
    INC (focused_decisions);
  solver->level++;
  assert (solver->level != INVALID_LEVEL);
  const unsigned idx = kissat_next_decision_variable (solver);
  const value value = decide_phase (solver, idx);
  unsigned lit = LIT (idx);
  if (value < 0)
    lit = NOT (lit);
  kissat_push_frame (solver, lit);
  assert (solver->level < SIZE_STACK (solver->frames));
  LOG ("decide literal %s", LOGLIT (lit));
  kissat_assign_decision (solver, lit);
  STOP (decide);
}

void
kissat_internal_assume (kissat * solver, unsigned lit)
{
  assert (solver->unassigned);
  assert (!VALUE (lit));
  solver->level++;
  assert (solver->level != INVALID_LEVEL);
  kissat_push_frame (solver, lit);
  assert (solver->level < SIZE_STACK (solver->frames));
  LOG ("assuming literal %s", LOGLIT (lit));
  kissat_assign_decision (solver, lit);
}
