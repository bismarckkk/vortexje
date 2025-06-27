/************************************************
 * Copyright (c) 2025. Zuo Qingyu               *
 * All rights reserved.                         *
 ************************************************/

#include <iostream>
#include <fstream>
#include <vector>

#include <highfive/highfive.hpp>
#include <highfive/eigen.hpp>

#include "utils/GenConfigProvider.hpp"
#include "xdmf-surface-writer.hpp"

const char *Vortexje::XdmfSurfaceWriter::file_extension() const {
    return ".";
}

bool Vortexje::XdmfSurfaceWriter::write(const std::shared_ptr<Surface> &surface, const std::string &_filename,
                                        int node_offset, int panel_offset, const std::vector<std::string> &view_names,
                                        const std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> &view_data) {
    std::string h5filename = _filename + "h5";
    std::string h5 = h5filename;
    size_t last_slash = h5filename.rfind('/');
    if (last_slash != std::string::npos) {
        h5 = h5filename.substr(last_slash + 1);
    }

    std::string filename = _filename + "xmf";
    PLOG.debug("Surface {}: Saving to {}.", surface->id, filename);

    size_t last_underscore = _filename.rfind('_');
    size_t last_dot = _filename.rfind('.');

    int step_number = -1;
    if (last_underscore != std::string::npos && last_dot != std::string::npos && last_dot > last_underscore) {
        std::string step_number_str = _filename.substr(last_underscore + 1, last_dot - last_underscore - 1);
        step_number = std::stoi(step_number_str);
    } else {
        throw std::runtime_error("Invalid filename format: " + _filename);
    }

    HighFive::DataSetCreateProps props;
//    props.add(HighFive::Deflate(5));
//    props.add(HighFive::Shuffle());

    std::ofstream xdmf(filename);
    HighFive::File h5file(h5filename, HighFive::File::Truncate);
    std::vector<Eigen::Vector3d> nodes;
    for (int i = 0; i < surface->n_nodes(); i++) {
        nodes.push_back(surface->nodes[i]);
    }
    HighFive::DataSet dataset = h5file.createDataSet("X", nodes, props);
    dataset.write(nodes);

    std::vector<int> panel_nodes;
    for (const auto& panel: surface->panel_nodes) {
        if (panel.size() == 4) {
            panel_nodes.push_back(5);
        } else if (panel.size() == 3) {
            panel_nodes.push_back(4);
        } else {
            throw std::runtime_error("Invalid panel size: " + std::to_string(panel.size()));
        }
        for (int idx : panel) {
            panel_nodes.push_back(idx);
        }
    }

    HighFive::DataSet panel_dataset = h5file.createDataSet("P", panel_nodes, props);
    panel_dataset.write(panel_nodes);

    xdmf << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
         << "<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\" Version=\"3.0\">\n"
         << "    <Domain>\n"
         << "        <Grid Name=\"surface\" GridType=\"Uniform\">\n"
         << "            <Time Value=\"" << step_number << "\" />\n"
         << "            <Geometry Type=\"XYZ\">\n"
         << "                <DataItem DataType=\"Float\" Dimensions=\"" << nodes.size() << " 3\" Format=\"HDF\" Precision=\"8\">" << h5 << ":X</DataItem>\n"
         << "            </Geometry>\n"
         << "            <Topology Type=\"Mixed\" Dimensions=\"" << surface->n_panels() << "\">\n"
         << "                <DataItem DataType=\"Int\" Dimensions=\"" << panel_nodes.size() << "\" Format=\"HDF\" Precision=\"4\">" << h5 << ":P</DataItem>\n"
         << "            </Topology>\n";

    for (int k = 0; k < (int) view_names.size(); k++) {
        if (view_data[k].cols() == 1) {
            xdmf << "            <Attribute Name=\"" << view_names[k] << "\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
                 << "                <DataItem DataType=\"Float\" Dimensions=\"" << surface->n_panels()
                 << "\" Format=\"HDF\" Precision=\"8\">" << h5 << ":" << view_names[k] << "</DataItem>\n"
                 << "            </Attribute>\n";
            std::vector<double> data;
            for (int i = 0; i < surface->n_panels(); i++) {
                for (int j = 0; j < view_data[k].cols(); j++) {
                    data.push_back(view_data[k](i, j));
                }
            }
            HighFive::DataSet view_dataset = h5file.createDataSet(view_names[k], data, props);
            view_dataset.write(data);
        } else {
            xdmf << "            <Attribute Name=\"" << view_names[k] << "\" AttributeType=\"Vector\" Center=\"Cell\">\n"
                 << "                <DataItem DataType=\"Float\" Dimensions=\"" << surface->n_panels()
                 << " 3\" Format=\"HDF\" Precision=\"8\">" << h5 << ":" << view_names[k] << "</DataItem>\n"
                 << "            </Attribute>\n";
            std::vector<Eigen::VectorXd> data;
            for (int i = 0; i < surface->n_panels(); i++) {
                data.emplace_back(view_data[k].row(i));
            }
            HighFive::DataSet view_dataset = h5file.createDataSet(view_names[k], data, props);
            view_dataset.write(data);
        }
    }

    xdmf << "        </Grid>\n"
         << "    </Domain>\n"
         << "</Xdmf>\n";
    xdmf.close();

    // Done:
    return true;
}
