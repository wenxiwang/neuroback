
#include "list.h"
#include <assert.h>
#include <stdio.h>


list create_list() {
	list l;
	l.head = NULL;
	l.tail = NULL;
	l.size = 0;
	return l;
}

void add_node(kissat * solver, list* l, unsigned var_idx) {
	size_t bytes = sizeof (node);
  	node* n = kissat_malloc (solver, bytes);
	n->var_idx = var_idx;
	n->next = NULL;
	
	if(l->size > 0) {
		l->tail->next = n;
		l->tail = n;	
		l->size += 1;
	}
	else {
		l->tail = n;	
		l->head = n;
		l->size += 1;
	}
}

// prev_node should be NULL, when the delete_node is the first in the list
// prev_condition: delete_node !=NULL
void delete_node(kissat * solver, list* l, node* prev_node, node* delete_node) {
	//assert(l->size > 0 && delete_node != NULL);
	if(l->head == delete_node) {
		assert(prev_node == NULL);
		l->head = delete_node->next;
	}
	if(l->tail == delete_node) {
		l->tail = prev_node;
	}
	l->size -= 1;
	
	if(prev_node)
		prev_node->next = delete_node->next;
	
	kissat_free(solver, delete_node, sizeof (node));	
}

void release_list(kissat* solver, list* l) {

	node* current_node = l->head;
	node* prev_node = NULL;

	while(current_node !=NULL) {
		
		delete_node(solver, l, prev_node, current_node);
		if(prev_node)
			current_node = prev_node->next;
		else
			current_node = NULL;	
	}
}

void print_list(list* l) {
	if(l->size == 0) {
		printf("list is empty\n");
	} 
	else {
		node* n = l->head;
		while(1) {
			printf("%u\n", n->var_idx);
			if(n == l->tail) {
				break;
			}
			n = n->next;
		}
		
	}
}
