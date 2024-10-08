//
// Vortexje -- Body.
//
// Copyright (C) 2012 - 2014 Baayen & Heinz GmbH.
//
// Authors: Jorn Baayen <jorn.baayen@baayen-heinz.com>
//

#include <vortexje/body.hpp>

#include <iostream>
#include <assert.h>

using namespace std;
using namespace Eigen;
using namespace Vortexje;

/**
   Constructs a new Body.
   
   @param[in]   id   Name for this body.
*/
Body::Body(const string &id) : id(id)
{
    // Initialize kinematics:
    position = Vector3d(0, 0, 0);
    velocity = Vector3d(0, 0, 0);
    
    attitude = Quaterniond(1, 0, 0, 0);
    rotational_velocity = Vector3d(0, 0, 0);
}

/**
   Body destructor.
*/
Body::~Body()
{
}

/**
   Adds a non-lifting surface to this body.
   
   @param[in]   non_lifting_surface   Non-lifting surface.
*/
void
Body::add_non_lifting_surface(std::shared_ptr<Surface> non_lifting_surface)
{
    non_lifting_surfaces.push_back(shared_ptr<SurfaceData>(new SurfaceData(non_lifting_surface)));
}

/**
   Adds a lifting surface to this body.
   
   @param[in]   lifting_surface   Lifting surface.
*/
void
Body::add_lifting_surface(std::shared_ptr<LiftingSurface> lifting_surface)
{
    shared_ptr<Wake> wake(new Wake(lifting_surface));
    
    add_lifting_surface(lifting_surface, wake);
}

/**
   Adds a lifting surface and its wake to this body.
   
   @param[in]   lifting_surface   Lifting surface.
   @param[in]   wake              Wake.
*/
void
Body::add_lifting_surface(std::shared_ptr<LiftingSurface> lifting_surface, std::shared_ptr<Wake> wake)
{
    lifting_surfaces.push_back(shared_ptr<LiftingSurfaceData>(new LiftingSurfaceData(lifting_surface, wake)));
}

/**
   Creates an across-surface neighborhood relationship between two panels.  This operation
   is also known as stitching.
   
   @param[in]   surface_a   First reference surface.
   @param[in]   panel_a     First reference panel.
   @param[in]   edge_a      First reference edge.
   @param[in]   surface_b   Second reference surface.
   @param[in]   panel_b     Second reference panel.
   @param[in]   edge_b      Second reference edge.
*/
void
Body::stitch_panels(std::shared_ptr<Surface> surface_a, int panel_a, int edge_a, std::shared_ptr<Surface> surface_b, int panel_b, int edge_b)
{
    // Add stitch from A to B:
    stitches[SurfacePanelEdge(surface_a, panel_a, edge_a)].push_back(SurfacePanelEdge(surface_b, panel_b, edge_b));

    // Add stitch from B to A:
    stitches[SurfacePanelEdge(surface_b, panel_b, edge_b)].push_back(SurfacePanelEdge(surface_a, panel_a, edge_a));
}

/**
   Lists both in-surface and across-surface (stitched) neighbors of the given panel.
   
   @param[in]   surface   Reference surface.
   @param[in]   panel     Reference panel.
   
   @returns List of in-surface and across-surface panel neighbors.
*/
vector<Body::SurfacePanelEdge>
Body::panel_neighbors(const std::shared_ptr<Surface> &surface, int panel) const
{
    vector<SurfacePanelEdge> neighbors;
    
    // List in-surface neighbors:
    for (int i = 0; i < (int) surface->panel_nodes[panel].size(); i++) {
        vector<pair<int, int> > &edge_neighbors = surface->panel_neighbors[panel][i];
        vector<pair<int, int> >::const_iterator it;
        for (it = edge_neighbors.begin(); it != edge_neighbors.end(); it++)
            neighbors.push_back(SurfacePanelEdge(surface, it->first, it->second));    
    }
    
    // List stitches:
    for (int i = 0; i < (int) surface->panel_nodes[panel].size(); i++) {
        map<SurfacePanelEdge, vector<SurfacePanelEdge>, CompareSurfacePanelEdge>::const_iterator it =
            stitches.find(SurfacePanelEdge(surface, panel, i));
        if (it != stitches.end()) {
            vector<SurfacePanelEdge> edge_stitches = it->second;
            vector<SurfacePanelEdge>::const_iterator sit;
            for (sit = edge_stitches.begin(); sit != edge_stitches.end(); sit++)
                neighbors.push_back(*sit);
        }
    }
    
    // Done:
    return neighbors;
}

/**
   Lists both in-surface and across-surface (stitched) neighbors of the given panel and edge.
   
   @param[in]   surface   Reference surface.
   @param[in]   panel     Reference panel.
   @param[in]   edge      Reference edge.
   
   @returns List of in-surface and across-surface panel neighbors for the given edge.
*/
vector<Body::SurfacePanelEdge>
Body::panel_neighbors(const std::shared_ptr<Surface> &surface, int panel, int edge) const
{
    vector<SurfacePanelEdge> neighbors;
    
    // List in-surface neighbors:
    {
        vector<pair<int, int> > &edge_neighbors = surface->panel_neighbors[panel][edge];
        vector<pair<int, int> >::const_iterator it;
        for (it = edge_neighbors.begin(); it != edge_neighbors.end(); it++)
            neighbors.push_back(SurfacePanelEdge(surface, it->first, it->second));
    }
    
    // List stitches:
    {
        map<SurfacePanelEdge, vector<SurfacePanelEdge>, CompareSurfacePanelEdge>::const_iterator it =
            stitches.find(SurfacePanelEdge(surface, panel, edge));
        if (it != stitches.end()) {
            vector<SurfacePanelEdge> edge_stitches = it->second;
            vector<SurfacePanelEdge>::const_iterator sit;
            for (sit = edge_stitches.begin(); sit != edge_stitches.end(); sit++)
                neighbors.push_back(*sit);
        }
    }
    
    // Done:
    return neighbors;
}

/**
   Sets the linear position of this body.
   
   @param[in]   position   Linear position.
*/
void
Body::set_position(const Vector3d &position)
{
    // Compute differential translation:
    Vector3d translation = position - this->position;
       
    // Apply:
    vector<shared_ptr<SurfaceData> >::iterator si;
    for (si = non_lifting_surfaces.begin(); si != non_lifting_surfaces.end(); si++) {
        shared_ptr<SurfaceData> d = *si;
        
        d->surface->translate(translation);
    }
    
    vector<shared_ptr<LiftingSurfaceData> >::iterator lsi;
    for (lsi = lifting_surfaces.begin(); lsi != lifting_surfaces.end(); lsi++) {
        shared_ptr<LiftingSurfaceData> d = *lsi;
        
        d->surface->translate(translation);
        
        d->wake->translate_trailing_edge(translation);
    }
    
    // Update state:
    this->position = position;
}

/**
   Sets the attitude (orientation) of this body.
   
   @param[in]   attitude   Attitude (orientation) of this body, as normalized quaternion.
*/
void
Body::set_attitude(const Quaterniond &attitude)
{   
    // Compute differential transformation:
    Transform<double, 3, Affine> transformation = Translation<double, 3>(position) * attitude * this->attitude.inverse() * Translation<double, 3>(-position);
    
    // Apply:
    vector<shared_ptr<SurfaceData> >::iterator si;
    for (si = non_lifting_surfaces.begin(); si != non_lifting_surfaces.end(); si++) {
        shared_ptr<SurfaceData> d = *si;
        
        d->surface->transform(transformation);
    }
    
    vector<shared_ptr<LiftingSurfaceData> >::iterator lsi;
    for (lsi = lifting_surfaces.begin(); lsi != lifting_surfaces.end(); lsi++) {
        shared_ptr<LiftingSurfaceData> d = *lsi;
        
        d->surface->transform(transformation);
        
        d->wake->transform_trailing_edge(transformation);
    }
    
    // Update state:
    this->attitude = attitude;
}

/**
   Sets the linear velocity of this body.
   
   @param[in]   velocity   Linear velocity.
*/
void 
Body::set_velocity(const Vector3d &velocity)
{
    this->velocity = velocity;
}

/**
   Sets the rotational velocity of this body.
   
   @param[in]   rotational_velocity   Rotational velocity.
*/
void
Body::set_rotational_velocity(const Vector3d &rotational_velocity)
{
    this->rotational_velocity = rotational_velocity;
}

/**
   Computes the kinematic velocity of the given panel.
   
   @param[in]   surface   Surface, belonging to this body. 
   @param[in]   panel     Panel, belonging to this surface.
   
   @return The kinematic velocity.
*/
Vector3d
Body::panel_kinematic_velocity(const std::shared_ptr<Surface> &surface, int panel) const
{
    return surface->panel_velocity[panel] - surface->panel_velocity_inflow[panel];
}

Eigen::Vector3d Body::kinematic_velocity(const Vector3d &point) const {
    Vector3d vel = velocity + rotational_velocity.cross(point - position);
    return vel;
}
