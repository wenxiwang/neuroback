#include "allocate.h"
#include "kissat.h"

typedef struct node {
	unsigned var_idx;
	struct node* next;
} node;

typedef struct list {
	node* head;
	node* tail;
	int size;	
} list;

extern list create_list();
extern void add_node(kissat *, list*, unsigned );
extern void delete_node(kissat * , list* , node* , node* );
extern void print_list(list* );

