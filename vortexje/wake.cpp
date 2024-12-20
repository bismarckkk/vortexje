//
// Vortexje -- Wake
//
// Copyright (C) 2012 - 2014 Baayen & Heinz GmbH.
//
// Authors: Jorn Baayen <jorn.baayen@baayen-heinz.com>
//

#include <iostream>

#include <vortexje/wake.hpp>
#include <vortexje/spline.h>

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

Vector3d vortex_line_unit_velocity(const Eigen::Vector3d &x, const Eigen::Vector3d &node_a, const Eigen::Vector3d &node_b)
{
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
        return {0, 0, 0};

    double r = sqrt(r_1xr_2_sqnorm) / r_0_norm;
    if (r < Parameters::wake_vortex_core_radius) {
        // Rankine vortex core segment:
        return r_1xr_2 / (r_0_norm * pow(Parameters::wake_vortex_core_radius, 2))
               * (r_0 / r_0_norm).dot(r_1 / r_1_norm - r_2 / r_2_norm);

    } else {
        // Free vortex segment:
        return r_1xr_2 / r_1xr_2_sqnorm * r_0.dot(r_1 / r_1_norm - r_2 / r_2_norm);

    }
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
        
        velocity += vortex_line_unit_velocity(x, node_a, node_b);
    }

    return one_over_4pi * velocity;
}

Vector3d Wake::vortex_line_velocity(const Vector3d &x) const {
    if (last_doublet_coefficients.empty()) {
        return {0, 0, 0};
    }
    Vector3d velocity(0, 0, 0);
    for (int i = 0; i < lifting_surface->n_spanwise_panels(); i++) {
        velocity += vortex_line_unit_velocity(x, nodes[i], nodes[i + 1]) * last_doublet_coefficients[i];
    }
    return -one_over_4pi * velocity;
}

Eigen::Vector3d calcVortex(
    const Eigen::Vector3d& p01,
    const Eigen::Vector3d& p10,
    const Eigen::Vector3d& p12,
    const Eigen::Vector3d& p21,
    double mu01,
    double mu10,
    double mu12,
    double mu21
) {
    Eigen::Vector3d su = p21 - p01;
    double lu = su.norm();
    Eigen::Vector3d eu = su / lu;
    double dmudu = (mu21 - mu01) / lu;

    Eigen::Vector3d s2 = p12 - p10;
    double l2 = s2.norm();
    Eigen::Vector3d e2 = s2 / l2;
    double dmud2 = (mu12 - mu10) / l2;

    Eigen::Vector3d ev = e2 - eu.dot(e2) * eu;
    ev.normalize();
    double dmudv = (dmud2 - eu.dot(e2) * dmudu) / ev.dot(e2);

    return (su.cross(s2)).norm() * (dmudu * ev - dmudv * eu);
}

std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> Wake::get_vortex_particles() {
    if (n_panels() / lifting_surface->n_spanwise_panels() < 2) {
        return std::make_pair(std::vector<Eigen::Vector3d>(), std::vector<Eigen::Vector3d>());
    }
    if (last_doublet_coefficients.empty()) {
        last_doublet_coefficients = std::vector<double>(lifting_surface->n_spanwise_panels(), 0);
        u0_v = std::vector<double>(lifting_surface->n_spanwise_nodes() + 1, 0);
        u0_x = std::vector<double>(lifting_surface->n_spanwise_nodes() + 1, 0);
        u0_y = std::vector<double>(lifting_surface->n_spanwise_nodes() + 1, 0);
        u0_z = std::vector<double>(lifting_surface->n_spanwise_nodes() + 1, 0);
        u0_mu = std::vector<double>(lifting_surface->n_spanwise_nodes() + 1, 0);
        u05_v = std::vector<double>(lifting_surface->n_spanwise_nodes() + 1, 0);
        u05_x = std::vector<double>(lifting_surface->n_spanwise_nodes() + 1, 0);
        u05_y = std::vector<double>(lifting_surface->n_spanwise_nodes() + 1, 0);
        u05_z = std::vector<double>(lifting_surface->n_spanwise_nodes() + 1, 0);
        u05_mu = std::vector<double>(lifting_surface->n_spanwise_nodes() + 1, 0);
        u1_v = std::vector<double>(lifting_surface->n_spanwise_nodes() + 1, 0);
        u1_x = std::vector<double>(lifting_surface->n_spanwise_nodes() + 1, 0);
        u1_y = std::vector<double>(lifting_surface->n_spanwise_nodes() + 1, 0);
        u1_z = std::vector<double>(lifting_surface->n_spanwise_nodes() + 1, 0);
        u1_mu = std::vector<double>(lifting_surface->n_spanwise_nodes() + 1, 0);
    }

    Eigen::Vector3d last_point_0 = nodes[lifting_surface->n_spanwise_nodes()];
    Eigen::Vector3d last_point_1 = nodes[0];
    Eigen::Vector3d last_point_05 = (last_point_0 + last_point_1) / 2;
    u0_x[0] = last_point_0.x();
    u0_y[0] = last_point_0.y();
    u0_z[0] = last_point_0.z();
    u05_x[0] = last_point_05.x();
    u05_y[0] = last_point_05.y();
    u05_z[0] = last_point_05.z();
    u1_x[0] = last_point_1.x();
    u1_y[0] = last_point_1.y();
    u1_z[0] = last_point_1.z();

    for (int i = 0; i < lifting_surface->n_spanwise_panels(); i++) {
        Eigen::Vector3d p0 = (nodes[i + lifting_surface->n_spanwise_nodes()] + nodes[i + lifting_surface->n_spanwise_nodes() + 1]) / 2;
        u0_x[i + 1] = p0.x();
        u0_y[i + 1] = p0.y();
        u0_z[i + 1] = p0.z();
        u0_v[i + 1] = u0_v[i] + (p0 - last_point_0).norm();
        u0_mu[i + 1] = (doublet_coefficients[i + lifting_surface->n_spanwise_panels()] + doublet_coefficients[i]) / 2;
        last_point_0 = p0;

        Eigen::Vector3d p05 = panel_collocation_points[0][i];
        u05_x[i + 1] = p05.x();
        u05_y[i + 1] = p05.y();
        u05_z[i + 1] = p05.z();
        u05_v[i + 1] = u05_v[i] + (p05 - last_point_05).norm();
        u05_mu[i + 1] = doublet_coefficients[i];
        last_point_05 = p05;

        Eigen::Vector3d p1 = (nodes[i] + nodes[i + 1]) / 2;
        u1_x[i + 1] = p1.x();
        u1_y[i + 1] = p1.y();
        u1_z[i + 1] = p1.z();
        u1_v[i + 1] = u1_v[i] + (p1 - last_point_1).norm();
        u1_mu[i + 1] = (doublet_coefficients[i] + last_doublet_coefficients[i]) / 2;
        last_point_1 = p1;
    }

    Eigen::Vector3d p0_end = nodes[lifting_surface->n_spanwise_nodes() + lifting_surface->n_spanwise_panels()];
    Eigen::Vector3d p1_end = nodes[lifting_surface->n_spanwise_panels()];
    Eigen::Vector3d p05_end = (p0_end + p1_end) / 2;
    u0_x[lifting_surface->n_spanwise_nodes()] = p0_end.x();
    u0_y[lifting_surface->n_spanwise_nodes()] = p0_end.y();
    u0_z[lifting_surface->n_spanwise_nodes()] = p0_end.z();
    u0_v[lifting_surface->n_spanwise_nodes()] = u0_v[lifting_surface->n_spanwise_panels()] + (p0_end - last_point_0).norm();
    u05_x[lifting_surface->n_spanwise_nodes()] = p05_end.x();
    u05_y[lifting_surface->n_spanwise_nodes()] = p05_end.y();
    u05_z[lifting_surface->n_spanwise_nodes()] = p05_end.z();
    u05_v[lifting_surface->n_spanwise_nodes()] = u05_v[lifting_surface->n_spanwise_panels()] + (p05_end - last_point_05).norm();
    u1_x[lifting_surface->n_spanwise_nodes()] = p1_end.x();
    u1_y[lifting_surface->n_spanwise_nodes()] = p1_end.y();
    u1_z[lifting_surface->n_spanwise_nodes()] = p1_end.z();
    u1_v[lifting_surface->n_spanwise_nodes()] = u1_v[lifting_surface->n_spanwise_panels()] + (p1_end - last_point_1).norm();

    tk::spline s0_x(u0_v, u0_x);
    tk::spline s0_y(u0_v, u0_y);
    tk::spline s0_z(u0_v, u0_z);
    tk::spline s0_mu(u0_v, u0_mu);
    tk::spline s05_x(u05_v, u05_x);
    tk::spline s05_y(u05_v, u05_y);
    tk::spline s05_z(u05_v, u05_z);
    tk::spline s05_mu(u05_v, u05_mu);
    tk::spline s1_x(u1_v, u1_x);
    tk::spline s1_y(u1_v, u1_y);
    tk::spline s1_z(u1_v, u1_z);
    tk::spline s1_mu(u1_v, u1_mu);

    double v0_max = u0_v.back();
    double v05_max = u05_v.back();
    double v1_max = u1_v.back();

    double v0_gap = v0_max / (double)lifting_surface->vpNumberPerStep;
    double v05_gap = v05_max / (double)lifting_surface->vpNumberPerStep;
    double v1_gap = v1_max / (double)lifting_surface->vpNumberPerStep;

    // 此处开始两位尾数分别代表u, v上的坐标
    // 对于u，上游边界为0，下游边界为2
    // 对于v，当前分段底部为0，顶部为2
    Eigen::Vector3d p00 = Eigen::Vector3d(s0_x(0), s0_y(0), s0_z(0));
    Eigen::Vector3d p10 = Eigen::Vector3d(s05_x(0), s05_y(0), s05_z(0));
    Eigen::Vector3d p20 = Eigen::Vector3d(s1_x(0), s1_y(0), s1_z(0));
    double mu00 = s0_mu(0);
    double mu10 = s05_mu(0);
    double mu20 = s1_mu(0);

    int vNumber = lifting_surface->vpNumberPerStep * lifting_surface->vlNumber;
    std::vector<Eigen::Vector3d> strength(vNumber, Eigen::Vector3d(0, 0, 0));
    std::vector<Eigen::Vector3d> position(vNumber, Eigen::Vector3d(0, 0, 0));

    for (int i = 0; i < lifting_surface->vpNumberPerStep; i++) {
        double v01 = v0_gap * ((double)i + 0.5);
        double v11 = v05_gap * ((double)i + 0.5);
        double v21 = v1_gap * ((double)i + 0.5);
        double v02 = v01 + v0_gap / 2;
        double v12 = v11 + v05_gap / 2;
        double v22 = v21 + v1_gap / 2;

        Eigen::Vector3d p01 = Eigen::Vector3d(s0_x(v01), s0_y(v01), s0_z(v01));
        Eigen::Vector3d p11 = Eigen::Vector3d(s05_x(v11), s05_y(v11), s05_z(v11));
        Eigen::Vector3d p21 = Eigen::Vector3d(s1_x(v21), s1_y(v21), s1_z(v21));

        Eigen::Vector3d p02 = Eigen::Vector3d(s0_x(v02), s0_y(v02), s0_z(v02));
        Eigen::Vector3d p12 = Eigen::Vector3d(s05_x(v12), s05_y(v12), s05_z(v12));
        Eigen::Vector3d p22 = Eigen::Vector3d(s1_x(v22), s1_y(v22), s1_z(v22));

        double mu01 = s0_mu(v01);
        double mu11 = s05_mu(v11);
        double mu21 = s1_mu(v21);

        double mu02 = s0_mu(v02);
        double mu12 = s05_mu(v12);
        double mu22 = s1_mu(v22);

        Eigen::Vector3d _p01 = p01;
        double _mu01 = mu01;
        double u_gap = 1. / lifting_surface->vlNumber;

        // _开头的变量的后缀的u坐标0代表当前网格左边界，2代表右边界，与上述描述不同
        for (int j = 0; j < lifting_surface->vlNumber; j++) {
            double u1 = u_gap * (j + 0.5);
            double u2 = u_gap * (j + 1);
            Eigen::Vector3d _p10, _p12, _p21;
            double _mu10, _mu12, _mu21;

            if (u1 < 0.5) {
                _p10 = p00 + (p10 - p00) * u1 * 2;
                _mu10 = mu00 + (mu10 - mu00) * u1 * 2;
                _p12 = p02 + (p12 - p02) * u1 * 2;
                _mu12 = mu02 + (mu12 - mu02) * u1 * 2;
            } else {
                _p10 = p10 + (p20 - p10) * (u1 - 0.5) * 2;
                _mu10 = mu10 + (mu20 - mu10) * (u1 - 0.5) * 2;
                _p12 = p12 + (p22 - p12) * (u1 - 0.5) * 2;
                _mu12 = mu12 + (mu22 - mu12) * (u1 - 0.5) * 2;
            }

            if (u2 < 0.5) {
                _p21 = p01 + (p11 - p01) * u2 * 2;
                _mu21 = mu01 + (mu11 - mu01) * u2 * 2;
            } else {
                _p21 = p11 + (p21 - p11) * (u2 - 0.5) * 2;
                _mu21 = mu11 + (mu21 - mu11) * (u2 - 0.5) * 2;
            }

            strength[i * lifting_surface->vlNumber + j] = calcVortex(_p01, _p10, _p12, _p21, _mu01, _mu10, _mu12, _mu21);
            position[i * lifting_surface->vlNumber + j] = (_p01 + _p10 + _p12 + _p21) / 4;

            _p01 = _p21;
            _mu01 = _mu21;
        }

        p00 = p02;
        p10 = p12;
        p20 = p22;
        mu00 = mu02;
        mu10 = mu12;
        mu20 = mu22;
    }

    for (int i = 0; i < lifting_surface->n_spanwise_panels(); i++) {
        last_doublet_coefficients[i] = doublet_coefficients[i];
    }

/*
 *
 */

//    std::vector<double> dudx(lifting_surface->n_spanwise_panels(), 0);
//    std::vector<double> dudy(lifting_surface->n_spanwise_panels(), 0);
//    std::vector<Eigen::Vector3d> esx(lifting_surface->n_spanwise_panels(), Eigen::Vector3d(0, 0, 0));
//    std::vector<Eigen::Vector3d> esy(lifting_surface->n_spanwise_panels(), Eigen::Vector3d(0, 0, 0));
//    for (int i = 0; i < lifting_surface->n_spanwise_panels(); i++) {
//        Eigen::Vector3d p1 = (nodes[i] + nodes[i + 1]) / 2;
//        Eigen::Vector3d p2 = (nodes[i + lifting_surface->n_spanwise_nodes()] + nodes[i + lifting_surface->n_spanwise_nodes() + 1]) / 2;
//        esx[i] = p1 - p2;
//        dudx[i] = (doublet_coefficients[i + lifting_surface->n_spanwise_panels()] - last_doublet_coefficients[i]) / esx[i].norm() / 2;
//        esx[i] /= esx[i].norm();
//        last_doublet_coefficients[i] = doublet_coefficients[i];
//    }
//    for (int i = 0; i < lifting_surface->n_spanwise_panels(); i++) {
//        Eigen::Vector3d p1 = (nodes[i] + nodes[i + lifting_surface->n_spanwise_nodes()]) / 2;
//        Eigen::Vector3d p2 = (nodes[i + 1] + nodes[i + 1 + lifting_surface->n_spanwise_nodes()]) / 2;
//        Eigen::Vector3d es2 = p1 - p2;
//        double duds2;
//        if (i == 0) {
//            duds2 = doublet_coefficients[i + 1] / es2.norm() / 2;
//        } else if (i == lifting_surface->n_spanwise_panels() - 1) {
//            duds2 = -doublet_coefficients[i - 1] / es2.norm() / 2;
//        } else {
//            duds2 = (doublet_coefficients[i + 1] - doublet_coefficients[i - 1]) / es2.norm() / 2;
//        }
//        esy[i] = es2 - esx[i].dot(es2) * esx[i];
//        esy[i] /= esy[i].norm();
//        es2 /= es2.norm();
//        dudy[i] = (duds2 - esx[i].dot(es2) * dudx[i]) / esy[i].dot(es2);
//    }
//
//    std::vector<Eigen::Vector3d> strength(lifting_surface->n_spanwise_panels(), Eigen::Vector3d(0, 0, 0));
//    std::vector<Eigen::Vector3d> position(lifting_surface->n_spanwise_panels(), Eigen::Vector3d(0, 0, 0));
//    for (int i = 0; i < lifting_surface->n_spanwise_panels(); i++) {
//        strength[i] = panel_surface_areas[i] * (dudx[i] * esy[i] - dudy[i] * esx[i]);
//        position[i] = panel_collocation_points[0][i];
//    }

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

