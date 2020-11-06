//
// Created by xiongchenhui on 2020/6/16.
//

#ifndef TENSORFLOW_SEASTAR_PORT_MGR_H
#define TENSORFLOW_SEASTAR_PORT_MGR_H

#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "tensorflow/core/platform/env.h"
#include "third_party/seastar/include/seastar/net/dns.hh"
#include "third_party/seastar/include/seastar/core/future-util.hh"
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <fstream>

namespace tensorflow {


    class SeastarPortMgr {
    public:
        explicit SeastarPortMgr(const ServerDef& server_def) {
            ParseGrpcServerDef(server_def);
            // LoadEndpointMapForFile();
        }
        std::string GetSeastarIpPort(const std::string& job_name, int task_index);
        int GetLocalSeastarPort();

        std::string get_job_name() const { return job_name_; }

        seastar::future<> resolve_ips_update_cluster_spec();

    private:
        void ParseGrpcServerDef(const ServerDef& server_def);
        void LoadEndpointMapForFile();



    private:
        std::unordered_map<std::string, std::string> endpoint_grpc2seastar_;
        std::unordered_map<std::string, std::unordered_map<int, std::string>>
                grpc_cluster_spec_;
        std::string job_name_;
        int task_index_;
        std::string local_grpc_ip_port_;
    };
}

#endif //TENSORFLOW_SEASTAR_PORT_MGR_H
