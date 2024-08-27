//
// Vortexje -- Wake
//
// Copyright (C) 2012 - 2014 Baayen & Heinz GmbH.
//
// Authors: Jorn Baayen <jorn.baayen@baayen-heinz.com>
//

#include <iostream>

#include <vortexje/wake.hpp>

using namespace std;
using namespace Eigen;
using namespace Vortexje;

static const double pi = 3.141592653589793238462643383279502884;

// Avoid having to divide by 4 pi all the time:
static const double one_over_4pi = 1.0 / (4 * pi);

/**
   Constructs an empty wake.
   
   @param[in]   lifting_surface   Associated lifting surface.
*/
Wake::Wake(shared_ptr<LiftingSurface> lifting_surface)
    : Surface(lifting_surface->id + string("_wake")), lifting_surface(lifting_surface)
{
}

/**
   Adds new layer of wake panels.
*/
void
Wake::add_layer()
{
    // Is this the first layer?
    bool first_layer;
    if (n_nodes() < lifting_surface->n_spanwise_nodes())
        first_layer = true;
    else
        first_layer = false;
        
    // Add layer of nodes at trailing edge, and add panels if necessary:
    for (int k = 0; k < lifting_surface->n_spanwise_nodes(); k++) {
        Vector3d new_point = lifting_surface->nodes[lifting_surface->trailing_edge_node(k)];
        
        int node = n_nodes();
        nodes.push_back(new_point);
        
        if (k > 0 && !first_layer) {
            vector<int> vertices;
            vertices.push_back(node - 1);
            vertices.push_back(node - 1 - lifting_surface->n_spanwise_nodes());
            vertices.push_back(node - lifting_surface->n_spanwise_nodes());
            vertices.push_back(node);
            
            int panel = n_panels();
            panel_nodes.push_back(vertices);
            panel_velocity.emplace_back(0, 0, 0);
            panel_velocity_inflow.emplace_back(0, 0, 0);
        
            vector<vector<pair<int, int> > > local_panel_neighbors;
            local_panel_neighbors.resize(vertices.size());
            panel_neighbors.push_back(local_panel_neighbors);
            
            shared_ptr<vector<int> > empty = make_shared<vector<int> >();
            node_panel_neighbors.push_back(empty);
            
            doublet_coefficients.push_back(0);
            
            compute_geometry(panel);

        } else {
            shared_ptr<vector<int> > empty = make_shared<vector<int> >();
            node_panel_neighbors.push_back(empty);
        }
    }
}

/**
   Translates the nodes of the trailing edge.
   
   @param[in]   translation   Translation vector.
*/
void
Wake::translate_trailing_edge(const Eigen::Vector3d &translation)
{
    if (n_nodes() < lifting_surface->n_spanwise_nodes())
        return;
        
    int k0;
    
    if (Parameters::convect_wake)
        k0 = n_nodes() - lifting_surface->n_spanwise_nodes();
    else
        k0 = 0;
        
    for (int k = k0; k < n_nodes(); k++)                
        nodes[k] += translation;
        
    if (Parameters::convect_wake)
        k0 = n_panels() - lifting_surface->n_spanwise_panels();
    else
        k0 = 0;
    
    for (int k = k0; k < n_panels(); k++)
        compute_geometry(k);
}

/**
   Transforms the nodes of the trailing edge.
   
   @param[in]   transformation   Affine transformation.
*/
void
Wake::transform_trailing_edge(const Eigen::Transform<double, 3, Eigen::Affine> &transformation)
{
    if (n_nodes() < lifting_surface->n_spanwise_nodes())
        return;
        
    int k0;
    
    if (Parameters::convect_wake)
        k0 = n_nodes() - lifting_surface->n_spanwise_nodes();
    else
        k0 = 0;
        
    for (int k = k0; k < n_nodes(); k++)                
        nodes[k] = transformation * nodes[k];
        
    if (Parameters::convect_wake)
        k0 = n_panels() - lifting_surface->n_spanwise_panels();
    else
        k0 = 0;
    
    for (int k = k0; k < n_panels(); k++)
        compute_geometry(k);
}

/**
   Updates any non-geometrical wake properties.  This method does nothing by default.
  
   @param[in]   dt   Time step size.
*/
void
Wake::update_properties(double dt)
{
}

/**
   Computes the velocity induced by a vortex ring of unit strength.
   
   @param[in]   x            Point at which the velocity is evaluated.
   @param[in]   this_panel   Panel on which the vortex ring is located.
   
   @returns Velocity induced by the vortex ring.
*/
Vector3d
Wake::vortex_ring_unit_velocity(const Eigen::Vector3d &x, int this_panel) const
{    
    Vector3d velocity(0, 0, 0);
    
    for (int i = 0; i < (int) panel_nodes[this_panel].size(); i++) {
        int previous_idx;
        if (i == 0)
            previous_idx = panel_nodes[this_panel].size() - 1;
        else
            previous_idx = i - 1;
            
        const Vector3d &node_a = nodes[panel_nodes[this_panel][previous_idx]];
        const Vector3d &node_b = nodes[panel_nodes[this_panel][i]];
        
        Vector3d r_0 = node_b - node_a;
        Vector3d r_1 = node_a - x;
        Vector3d r_2 = node_b - x;
        
        double r_0_norm = r_0.norm();
        double r_1_norm = r_1.norm();
        double r_2_norm = r_2.norm();
        
        Vector3d r_1xr_2 = r_1.cross(r_2);
        double r_1xr_2_sqnorm = r_1xr_2.squaredNorm();
        
        if (r_0_norm < Parameters::zero_threshold ||
            r_1_norm < Parameters::zero_threshold ||
            r_2_norm < Parameters::zero_threshold ||
            r_1xr_2_sqnorm < Parameters::zero_threshold)
            continue;

        double r = sqrt(r_1xr_2_sqnorm) / r_0_norm;
        if (r < Parameters::wake_vortex_core_radius) {
            // Rankine vortex core segment:
            velocity += r_1xr_2 / (r_0_norm * pow(Parameters::wake_vortex_core_radius, 2))
                * (r_0 / r_0_norm).dot(r_1 / r_1_norm - r_2 / r_2_norm);
                
        } else {
            // Free vortex segment:
            velocity += r_1xr_2 / r_1xr_2_sqnorm * r_0.dot(r_1 / r_1_norm - r_2 / r_2_norm);
            
        }
    }

    return one_over_4pi * velocity;
}

std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> Wake::get_vortex_particles() {
    if (n_panels() / lifting_surface->n_spanwise_panels() < 2) {
        return std::make_pair(std::vector<Eigen::Vector3d>(), std::vector<Eigen::Vector3d>());
    }
    if (last_doublet_coefficients.empty()) {
        last_doublet_coefficients = std::vector<double>(lifting_surface->n_spanwise_panels(), 0);
    }

    std::vector<double> dudx(lifting_surface->n_spanwise_panels(), 0);
    std::vector<double> dudy(lifting_surface->n_spanwise_panels(), 0);
    std::vector<Eigen::Vector3d> esx(lifting_surface->n_spanwise_panels(), Eigen::Vector3d(0, 0, 0));
    std::vector<Eigen::Vector3d> esy(lifting_surface->n_spanwise_panels(), Eigen::Vector3d(0, 0, 0));
    for (int i = 0; i < lifting_surface->n_spanwise_panels(); i++) {
        Eigen::Vector3d p1 = (nodes[i] + nodes[i + 1]) / 2;
        Eigen::Vector3d p2 = (nodes[i + lifting_surface->n_spanwise_nodes()] + nodes[i + lifting_surface->n_spanwise_nodes() + 1]) / 2;
        esx[i] = p1 - p2;
        dudx[i] = (doublet_coefficients[i + lifting_surface->n_spanwise_panels()] - last_doublet_coefficients[i]) / esx[i].norm() / 2;
        esx[i] /= esx[i].norm();
        last_doublet_coefficients[i] = doublet_coefficients[i];
    }
    for (int i = 0; i < lifting_surface->n_spanwise_panels(); i++) {
        Eigen::Vector3d p1 = (nodes[i] + nodes[i + lifting_surface->n_spanwise_nodes()]) / 2;
        Eigen::Vector3d p2 = (nodes[i + 1] + nodes[i + 1 + lifting_surface->n_spanwise_nodes()]) / 2;
        Eigen::Vector3d es2 = p1 - p2;
        double duds2;
        if (i == 0) {
            duds2 = doublet_coefficients[i + 1] / es2.norm() / 2;
        } else if (i == lifting_surface->n_spanwise_panels() - 1) {
            duds2 = -doublet_coefficients[i - 1] / es2.norm() / 2;
        } else {
            duds2 = (doublet_coefficients[i + 1] - doublet_coefficients[i - 1]) / es2.norm() / 2;
        }
        esy[i] = es2 - esx[i].dot(es2) * esx[i];
        esy[i] /= esy[i].norm();
        es2 /= es2.norm();
        dudy[i] = (duds2 - esx[i].dot(es2) * dudx[i]) / esy[i].dot(es2);
    }

    std::vector<Eigen::Vector3d> strength(lifting_surface->n_spanwise_panels(), Eigen::Vector3d(0, 0, 0));
    std::vector<Eigen::Vector3d> position(lifting_surface->n_spanwise_panels(), Eigen::Vector3d(0, 0, 0));
    for (int i = 0; i < lifting_surface->n_spanwise_panels(); i++) {
        strength[i] = panel_surface_areas[i] * (dudx[i] * esy[i] - dudy[i] * esx[i]);
        position[i] = panel_collocation_points[0][i];
    }

    remove_layer();

    return std::make_pair(strength, position);
}

void Wake::remove_layer() {
    if (n_panels() / lifting_surface->n_spanwise_panels() < 2) {
        return;
    }

    nodes.erase(nodes.begin(), nodes.begin() + lifting_surface->n_spanwise_nodes());
    panel_nodes.erase(panel_nodes.begin(), panel_nodes.begin() + lifting_surface->n_spanwise_panels());
    panel_velocity.erase(panel_velocity.begin(), panel_velocity.begin() + lifting_surface->n_spanwise_panels());
    panel_velocity_inflow.erase(panel_velocity_inflow.begin(), panel_velocity_inflow.begin() + lifting_surface->n_spanwise_panels());
    panel_neighbors.erase(panel_neighbors.begin(), panel_neighbors.begin() + lifting_surface->n_spanwise_panels());
    node_panel_neighbors.erase(node_panel_neighbors.begin(), node_panel_neighbors.begin() + lifting_surface->n_spanwise_nodes());
    doublet_coefficients.erase(doublet_coefficients.begin(), doublet_coefficients.begin() + lifting_surface->n_spanwise_panels());

    for (int i = 0; i < n_nodes(); i++) {
        for (auto& neighbor : *node_panel_neighbors[i]) {
            neighbor -= lifting_surface->n_spanwise_panels();
        }
    }
    for (int i = 0; i < n_panels(); i++) {
        for (auto& node : panel_nodes[i]) {
            node -= lifting_surface->n_spanwise_nodes();
        }
        for (auto& mother_edge : panel_neighbors[i]) {
            for (auto& neighbor : mother_edge) {
                neighbor.first -= lifting_surface->n_spanwise_panels();
            }
        }
    }
}

