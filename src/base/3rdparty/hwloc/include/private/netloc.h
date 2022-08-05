/*
 * Copyright © 2014 Cisco Systems, Inc.  All rights reserved.
 * Copyright © 2013-2014 University of Wisconsin-La Crosse.
 *                         All rights reserved.
 * Copyright © 2015-2017 Inria.  All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 * See COPYING in top-level directory.
 *
 * $HEADER$
 */

#ifndef _NETLOC_PRIVATE_H_
#define _NETLOC_PRIVATE_H_

#include <hwloc.h>
#include <netloc.h>
#include <netloc/uthash.h>
#include <netloc/utarray.h>
#include <private/autogen/config.h>

#define NETLOCFILE_VERSION 1

#ifdef NETLOC_SCOTCH
#include <stdint.h>
#include <scotch.h>
#define NETLOC_int SCOTCH_Num
#else
#define NETLOC_int int
#endif

/*
 * "Import" a few things from hwloc
 */
#define __netloc_attribute_unused __hwloc_attribute_unused
#define __netloc_attribute_malloc __hwloc_attribute_malloc
#define __netloc_attribute_const __hwloc_attribute_const
#define __netloc_attribute_pure __hwloc_attribute_pure
#define __netloc_attribute_deprecated __hwloc_attribute_deprecated
#define __netloc_attribute_may_alias __hwloc_attribute_may_alias
#define NETLOC_DECLSPEC HWLOC_DECLSPEC


/**********************************************************************
 * Types
 **********************************************************************/

/**
 * Definitions for Comparators
 * \sa These are the return values from the following functions:
 *     netloc_network_compare, netloc_dt_edge_t_compare, netloc_dt_node_t_compare
 */
typedef enum {
    NETLOC_CMP_SAME    =  0,  /**< Compared as the Same */
    NETLOC_CMP_SIMILAR = -1,  /**< Compared as Similar, but not the Same */
    NETLOC_CMP_DIFF    = -2   /**< Compared as Different */
} netloc_compare_type_t;

/**
 * Enumerated type for the various types of supported networks
 */
typedef enum {
    NETLOC_NETWORK_TYPE_ETHERNET    = 1, /**< Ethernet network */
    NETLOC_NETWORK_TYPE_INFINIBAND  = 2, /**< InfiniBand network */
    NETLOC_NETWORK_TYPE_INVALID     = 3  /**< Invalid network */
} netloc_network_type_t;

/**
 * Enumerated type for the various types of supported topologies
 */
typedef enum {
    NETLOC_TOPOLOGY_TYPE_INVALID = -1, /**< Invalid */
    NETLOC_TOPOLOGY_TYPE_TREE    = 1,  /**< Tree */
} netloc_topology_type_t;

/**
 * Enumerated type for the various types of nodes
 */
typedef enum {
    NETLOC_NODE_TYPE_HOST    = 0, /**< Host (a.k.a., network addressable endpoint - e.g., MAC Address) node */
    NETLOC_NODE_TYPE_SWITCH  = 1, /**< Switch node */
    NETLOC_NODE_TYPE_INVALID = 2  /**< Invalid node */
} netloc_node_type_t;

typedef enum {
    NETLOC_ARCH_TREE    =  0,  /* Fat tree */
} netloc_arch_type_t;


/* Pre declarations to avoid inter dependency problems */
/** \cond IGNORE */
struct netloc_topology_t;
typedef struct netloc_topology_t netloc_topology_t;
struct netloc_node_t;
typedef struct netloc_node_t netloc_node_t;
struct netloc_edge_t;
typedef struct netloc_edge_t netloc_edge_t;
struct netloc_physical_link_t;
typedef struct netloc_physical_link_t netloc_physical_link_t;
struct netloc_path_t;
typedef struct netloc_path_t netloc_path_t;

struct netloc_arch_tree_t;
typedef struct netloc_arch_tree_t netloc_arch_tree_t;
struct netloc_arch_node_t;
typedef struct netloc_arch_node_t netloc_arch_node_t;
struct netloc_arch_node_slot_t;
typedef struct netloc_arch_node_slot_t netloc_arch_node_slot_t;
struct netloc_arch_t;
typedef struct netloc_arch_t netloc_arch_t;
/** \endcond */

/**
 * \struct netloc_topology_t
 * \brief Netloc Topology Context
 *
 * An opaque data structure used to reference a network topology.
 *
 * \note Must be initialized with \ref netloc_topology_construct()
 */
struct netloc_topology_t {
    /** Topology path */
    char *topopath;
    /** Subnet ID */
    char *subnet_id;

    /** Node List */
    netloc_node_t *nodes; /* Hash table of nodes by physical_id */
    netloc_node_t *nodesByHostname; /* Hash table of nodes by hostname */

    netloc_physical_link_t *physical_links; /* Hash table with physcial links */

    /** Partition List */
    UT_array *partitions;

    /** Hwloc topology List */
    char *hwlocpath;
    UT_array *topos;
    hwloc_topology_t *hwloc_topos;

    /** Type of the graph */
    netloc_topology_type_t type;
};

/**
 * \brief Netloc Node Type
 *
 * Represents the concept of a node (a.k.a., vertex, endpoint) within a network
 * graph. This could be a server or a network switch. The \ref node_type parameter
 * will distinguish the exact type of node this represents in the graph.
 */
struct netloc_node_t {
    UT_hash_handle hh;       /* makes this structure hashable with physical_id */
    UT_hash_handle hh2;      /* makes this structure hashable with hostname */

    /** Physical ID of the node */
    char physical_id[20];

    /** Logical ID of the node (if any) */
    int logical_id;

    /** Type of the node */
    netloc_node_type_t type;

    /* Pointer to physical_links */
    UT_array *physical_links;

    /** Description information from discovery (if any) */
    char *description;

    /**
     * Application-given private data pointer.
     * Initialized to NULL, and not used by the netloc library.
     */
    void * userdata;

    /** Outgoing edges from this node */
    netloc_edge_t *edges;

    UT_array *subnodes; /* the group of nodes for the virtual nodes */

    netloc_path_t *paths;

    char *hostname;

    UT_array *partitions; /* index in the list from the topology */

    hwloc_topology_t hwlocTopo;
    int hwlocTopoIdx;
};

/**
 * \brief Netloc Edge Type
 *
 * Represents the concept of a directed edge within a network graph.
 *
 * \note We do not point to the netloc_node_t structure directly to
 * simplify the representation, and allow the information to more easily
 * be entered into the data store without circular references.
 * \todo JJH Is the note above still true?
 */
struct netloc_edge_t {
    UT_hash_handle hh;       /* makes this structure hashable */

    netloc_node_t *dest;

    int id;

    /** Pointers to the parent node */
    netloc_node_t *node;

    /* Pointer to physical_links */
    UT_array *physical_links;

    /** total gbits of the links */
    float total_gbits;

    UT_array *partitions; /* index in the list from the topology */

    UT_array *subnode_edges; /* for edges going to virtual nodes */

    struct netloc_edge_t *other_way;

    /**
     * Application-given private data pointer.
     * Initialized to NULL, and not used by the netloc library.
     */
    void * userdata;
};


struct netloc_physical_link_t {
    UT_hash_handle hh;       /* makes this structure hashable */

    int id; // TODO long long
    netloc_node_t *src;
    netloc_node_t *dest;
    int ports[2];
    char *width;
    char *speed;

    netloc_edge_t *edge;

    int other_way_id;
    struct netloc_physical_link_t *other_way;

    UT_array *partitions; /* index in the list from the topology */

    /** gbits of the link from speed and width */
    float gbits;

    /** Description information from discovery (if any) */
    char *description;
};

struct netloc_path_t {
    UT_hash_handle hh;       /* makes this structure hashable */
    char dest_id[20];
    UT_array *links;
};


/**********************************************************************
 *        Architecture structures
 **********************************************************************/
struct netloc_arch_tree_t {
    NETLOC_int num_levels;
    NETLOC_int *degrees;
    NETLOC_int *cost;
};

struct netloc_arch_node_t {
    UT_hash_handle hh;       /* makes this structure hashable */
    char *name; /* Hash key */
    netloc_node_t *node; /* Corresponding node */
    int idx_in_topo; /* idx with ghost hosts to have complete topo */
    int num_slots; /* it is not the real number of slots but the maximum slot idx */
    int *slot_idx; /* corresponding idx in slot_tree */
    int *slot_os_idx; /* corresponding os index for each leaf in tree */
    netloc_arch_tree_t *slot_tree; /* Tree built from hwloc */
    int num_current_slots; /* Number of PUs */
    NETLOC_int *current_slots; /* indices in the complete tree */
    int *slot_ranks; /* corresponding MPI rank for each leaf in tree */
};

struct netloc_arch_node_slot_t {
    netloc_arch_node_t *node;
    int slot;
};

struct netloc_arch_t {
    netloc_topology_t *topology;
    int has_slots; /* if slots are included in the architecture */
    netloc_arch_type_t type;
    union {
        netloc_arch_tree_t *node_tree;
        netloc_arch_tree_t *global_tree;
    } arch;
    netloc_arch_node_t *nodes_by_name;
    netloc_arch_node_slot_t *node_slot_by_idx; /* node_slot by index in complete topo */
    NETLOC_int num_current_hosts; /* if has_slots, host is a slot, else host is a node */
    NETLOC_int *current_hosts; /* indices in the complete topology */
};

/**********************************************************************
 * Topology Functions
 **********************************************************************/
/**
 * Allocate a topology handle.
 *
 * User is responsible for calling \ref netloc_detach on the topology handle.
 * The network parameter information is deep copied into the topology handle, so the
 * user may destruct the network handle after calling this function and/or reuse
 * the network handle.
 *
 * \returns NETLOC_SUCCESS on success
 * \returns NETLOC_ERROR upon an error.
 */
netloc_topology_t *netloc_topology_construct(char *path);

/**
 * Destruct a topology handle
 *
 * \param topology A valid pointer to a \ref netloc_topology_t handle created
 * from a prior call to \ref netloc_topology_construct.
 *
 * \returns NETLOC_SUCCESS on success
 * \returns NETLOC_ERROR upon an error.
 */
int netloc_topology_destruct(netloc_topology_t *topology);

int netloc_topology_find_partition_idx(netloc_topology_t *topology, char *partition_name);

int netloc_topology_read_hwloc(netloc_topology_t *topology, int num_nodes,
        netloc_node_t **node_list);

#define netloc_topology_iter_partitions(topology,partition) \
    for ((partition) = (char **)utarray_front(topology->partitions); \
            (partition) != NULL; \
            (partition) = (char **)utarray_next(topology->partitions, partition))

#define netloc_topology_iter_hwloctopos(topology,hwloctopo) \
    for ((hwloctopo) = (char **)utarray_front(topology->topos); \
            (hwloctopo) != NULL; \
            (hwloctopo) = (char **)utarray_next(topology->topos, hwloctopo))

#define netloc_topology_find_node(topology,node_id,node) \
    HASH_FIND_STR(topology->nodes, node_id, node)

#define netloc_topology_iter_nodes(topology,node,_tmp) \
    HASH_ITER(hh, topology->nodes, node, _tmp)

#define netloc_topology_num_nodes(topology) \
    HASH_COUNT(topology->nodes)

/*************************************************/


/**
 * Constructor for netloc_node_t
 *
 * User is responsible for calling the destructor on the handle.
 *
 * Returns
 *   A newly allocated pointer to the network information.
 */
netloc_node_t *netloc_node_construct(void);

/**
 * Destructor for netloc_node_t
 *
 * \param node A valid node handle
 *
 * Returns
 *   NETLOC_SUCCESS on success
 *   NETLOC_ERROR on error
 */
int netloc_node_destruct(netloc_node_t *node);

char *netloc_node_pretty_print(netloc_node_t* node);

#define netloc_node_get_num_subnodes(node) \
    utarray_len((node)->subnodes)

#define netloc_node_get_subnode(node,i) \
    (*(netloc_node_t **)utarray_eltptr((node)->subnodes, (i)))

#define netloc_node_get_num_edges(node) \
    utarray_len((node)->edges)

#define netloc_node_get_edge(node,i) \
    (*(netloc_edge_t **)utarray_eltptr((node)->edges, (i)))

#define netloc_node_iter_edges(node,edge,_tmp) \
    HASH_ITER(hh, node->edges, edge, _tmp)

#define netloc_node_iter_paths(node,path,_tmp) \
    HASH_ITER(hh, node->paths, path, _tmp)

#define netloc_node_is_host(node) \
    (node->type == NETLOC_NODE_TYPE_HOST)

#define netloc_node_is_switch(node) \
    (node->type == NETLOC_NODE_TYPE_SWITCH)

#define netloc_node_iter_paths(node, path,_tmp) \
    HASH_ITER(hh, node->paths, path, _tmp)

int netloc_node_is_in_partition(netloc_node_t *node, int partition);

/*************************************************/


/**
 * Constructor for netloc_edge_t
 *
 * User is responsible for calling the destructor on the handle.
 *
 * Returns
 *   A newly allocated pointer to the edge information.
 */
netloc_edge_t *netloc_edge_construct(void);

/**
 * Destructor for netloc_edge_t
 *
 * \param edge A valid edge handle
 *
 * Returns
 *   NETLOC_SUCCESS on success
 *   NETLOC_ERROR on error
 */
int netloc_edge_destruct(netloc_edge_t *edge);

char * netloc_edge_pretty_print(netloc_edge_t* edge);

void netloc_edge_reset_uid(void);

int netloc_edge_is_in_partition(netloc_edge_t *edge, int partition);

#define netloc_edge_get_num_links(edge) \
    utarray_len((edge)->physical_links)

#define netloc_edge_get_link(edge,i) \
    (*(netloc_physical_link_t **)utarray_eltptr((edge)->physical_links, (i)))

#define netloc_edge_get_num_subedges(edge) \
    utarray_len((edge)->subnode_edges)

#define netloc_edge_get_subedge(edge,i) \
    (*(netloc_edge_t **)utarray_eltptr((edge)->subnode_edges, (i)))

/*************************************************/


/**
 * Constructor for netloc_physical_link_t
 *
 * User is responsible for calling the destructor on the handle.
 *
 * Returns
 *   A newly allocated pointer to the physical link information.
 */
netloc_physical_link_t * netloc_physical_link_construct(void);

/**
 * Destructor for netloc_physical_link_t
 *
 * Returns
 *   NETLOC_SUCCESS on success
 *   NETLOC_ERROR on error
 */
int netloc_physical_link_destruct(netloc_physical_link_t *link);

char * netloc_link_pretty_print(netloc_physical_link_t* link);

/*************************************************/


netloc_path_t *netloc_path_construct(void);
int netloc_path_destruct(netloc_path_t *path);


/**********************************************************************
 *        Architecture functions
 **********************************************************************/

netloc_arch_t * netloc_arch_construct(void);

int netloc_arch_destruct(netloc_arch_t *arch);

int netloc_arch_build(netloc_arch_t *arch, int add_slots);

int netloc_arch_set_current_resources(netloc_arch_t *arch);

int netloc_arch_set_global_resources(netloc_arch_t *arch);

int netloc_arch_node_get_hwloc_info(netloc_arch_node_t *arch);

void netloc_arch_tree_complete(netloc_arch_tree_t *tree, UT_array **down_degrees_by_level,
        int num_hosts, int **parch_idx);

NETLOC_int netloc_arch_tree_num_leaves(netloc_arch_tree_t *tree);


/**********************************************************************
 *        Access functions of various elements of the topology
 **********************************************************************/

#define netloc_get_num_partitions(object) \
    utarray_len((object)->partitions)

#define netloc_get_partition(object,i) \
    (*(int *)utarray_eltptr((object)->partitions, (i)))


#define netloc_path_iter_links(path,link) \
    for ((link) = (netloc_physical_link_t **)utarray_front(path->links); \
            (link) != NULL; \
            (link) = (netloc_physical_link_t **)utarray_next(path->links, link))

/**********************************************************************
 *        Misc functions
 **********************************************************************/

/**
 * Decode the network type
 *
 * \param net_type A valid member of the \ref netloc_network_type_t type
 *
 * \returns NULL if the type is invalid
 * \returns A string for that \ref netloc_network_type_t type
 */
static inline const char * netloc_network_type_decode(netloc_network_type_t net_type) {
    if( NETLOC_NETWORK_TYPE_ETHERNET == net_type ) {
        return "ETH";
    }
    else if( NETLOC_NETWORK_TYPE_INFINIBAND == net_type ) {
        return "IB";
    }
    else {
        return NULL;
    }
}

/**
 * Decode the node type
 *
 * \param node_type A valid member of the \ref netloc_node_type_t type
 *
 * \returns NULL if the type is invalid
 * \returns A string for that \ref netloc_node_type_t type
 */
static inline const char * netloc_node_type_decode(netloc_node_type_t node_type) {
    if( NETLOC_NODE_TYPE_SWITCH == node_type ) {
        return "SW";
    }
    else if( NETLOC_NODE_TYPE_HOST == node_type ) {
        return "CA";
    }
    else {
        return NULL;
    }
}

ssize_t netloc_line_get(char **lineptr, size_t *n, FILE *stream);

char *netloc_line_get_next_token(char **string, char c);

int netloc_build_comm_mat(char *filename, int *pn, double ***pmat);

#define STRDUP_IF_NOT_NULL(str) (NULL == str ? NULL : strdup(str))
#define STR_EMPTY_IF_NULL(str) (NULL == str ? "" : str)


#endif // _NETLOC_PRIVATE_H_
