//
// Created by xiongchenhui on 2020/6/16.
//

#include "tensorflow/contrib/seastar/seastar_port_mgr.h"
#include "third_party/seastar/include/seastar/core/sleep.hh"
#include <boost/xpressive/xpressive_dynamic.hpp>

namespace tensorflow {

    namespace {

        const char* kEndpointMapFile = ".endpoint_map";
        const std::map<int,int> portMap = {
                std::make_pair(2222,9999),
                std::make_pair(2223,9993),
                std::make_pair(3333,8888)
        };

    }  // namespace

    std::string SeastarPortMgr::GetSeastarIpPort(const std::string& job_name, int task_index) {
        const auto it_job = grpc_cluster_spec_.find(job_name);
        if (it_job == grpc_cluster_spec_.end()) {
            LOG(FATAL) << "Job name: " << job_name
                       << " does not exist in cluster spec.";
        }
        const auto& task_map = it_job->second;

        const auto it_task = task_map.find(task_index);
        if (it_task == task_map.end()) {
            LOG(FATAL) << "Job name: " << job_name << ", task index: " << task_index
                       << " does not exist in cluster spec.";
        }
        const std::string& grpc_ip_port = it_task->second;

        std::vector<std::string> vec = str_util::Split(grpc_ip_port, ":");
        CHECK_EQ(vec.size(), 2);
        int local_grpc_port = -1;
        strings::safe_strto32(vec[1], &local_grpc_port);
        CHECK_GT(local_grpc_port, 0);
        const auto it_seastar = portMap.find(local_grpc_port);
        if (it_seastar == portMap.end()) {
            LOG(FATAL) << "Seastar ip and port not found for job name: " << job_name
                       << "task index: " << task_index << ".";
        }

        std::string seastar_ip_port = vec[0]+":"+std::to_string(it_seastar->second);

        return seastar_ip_port;
    }

    int SeastarPortMgr::GetLocalSeastarPort() {
        std::vector<std::string> vec = str_util::Split(local_grpc_ip_port_, ":");
        CHECK_EQ(vec.size(), 2);

        int local_grpc_port = -1;
        strings::safe_strto32(vec[1], &local_grpc_port);
        CHECK_GT(local_grpc_port, 0);
        const auto it = portMap.find(local_grpc_port);
        if (it == portMap.end()) {
            LOG(FATAL) << "Seastar port not found for job name: " << job_name_
                       << "task index: " << task_index_ << ".";
        }

        return it->second;
    }

    void SeastarPortMgr::ParseGrpcServerDef(const ServerDef& server_def) {
        job_name_ = server_def.job_name();
        task_index_ = server_def.task_index();

        for (const auto& job : server_def.cluster().job()) {
            auto& task_map = grpc_cluster_spec_[job.name()];
            for (const auto& task : job.tasks()) {
                task_map[task.first] = task.second;
                if (job.name() == job_name_ && task.first == task_index_) {
                    local_grpc_ip_port_ = task.second;
                }
            }
        }

        if (local_grpc_ip_port_.empty()) {
            LOG(FATAL) << "Job name: " << job_name_ << ", task index: " << task_index_
                       << " not found in cluter spec.";
        }
    }

    seastar::future<> SeastarPortMgr::resolve_ips_update_cluster_spec() {
        using namespace boost::xpressive;
        using namespace seastar;
        using namespace seastar::net;
        std::unordered_map<std::string, std::unordered_map<int, std::string>> update_clusters = grpc_cluster_spec_;
        cregex reg_ip = cregex::compile("(25[0-4]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[1-9])[.](25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])[.](25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])[.](25[0-4]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[1-9])");
        std::vector<seastar::future<net::inet_address> > future_vec;
        std::vector<std::shared_ptr<net::dns_resolver>> dns_resolver_vec;
        std::unordered_map<std::string, int> service_names;
        int index = 0;
        for (auto it = grpc_cluster_spec_.begin(); it != grpc_cluster_spec_.end(); it++) {
            auto job_name = it->first;
            std::unordered_map<int, std::string>& task_map = it->second;
            for (auto task_it = task_map.begin(); task_it != task_map.end(); task_it++) {
                std::vector<std::string> vec = str_util::Split(task_it->second, ":");
                CHECK_EQ(vec.size(), 2);
                if (regex_match(vec[0].c_str(), reg_ip)) {
                    continue;
                }
                auto d = std::make_shared<dns_resolver>();
                service_names.insert({task_it->second,index++});
                future_vec.push_back(d->get_host_by_name(std::move(vec[0]), inet_address::family::INET).then([](hostent h) {
                    return make_ready_future<inet_address>(h.addr_list.front());
                }));
                dns_resolver_vec.push_back(d);
            }
        }

        return when_all_succeed(future_vec.begin(),future_vec.end()).then([this,dns_resolver_vec,service_names](auto vals) {
            int n = vals.size();
            LOG(ERROR) << "address:" << vals.size();
            for (int i=0; i<n; i++) {
                LOG(ERROR) << "ip:" << vals[i];
            }

            for (auto it = this->grpc_cluster_spec_.begin(); it != this->grpc_cluster_spec_.end(); it++) {
                auto job_name = it->first;
                std::unordered_map<int, std::string>& task_map = it->second;
                for (auto task_it = task_map.begin(); task_it != task_map.end(); task_it++) {
                    std::vector<std::string> vec = str_util::Split(task_it->second, ":");
                    CHECK_EQ(vec.size(), 2);
                    auto index_it = service_names.find(task_it->second);
                    if (index_it != service_names.end()) {
                        auto real_ip = vals[index_it->second];
                        std::stringstream ss;
                        ss << real_ip;
                        task_it->second = ss.str()+ ":" + vec[1];
                        LOG(ERROR) << "real ip:" << task_it->second;
                    }
                }
            }
            for (int i=0; i<dns_resolver_vec.size(); i++) {
                auto cf = dns_resolver_vec[i]->close().then([i]{LOG(ERROR) << "dns_resolver close at index:" << i;});
            }


        }).handle_exception([this,dns_resolver_vec] (auto excp) {
            try {
                std::rethrow_exception(excp);
            } catch (const std::exception& e) {
                LOG(ERROR) << "resolve_ips_update_cluster_spec exception:" << e.what();
            }
            for (int i=0; i<dns_resolver_vec.size(); i++) {
                auto cf = dns_resolver_vec[i]->close().then([i]{LOG(ERROR) << "dns_resolver close at index:" << i;});
            }
            using namespace std::chrono_literals;
            return seastar::sleep(1s).then([this] { return this->resolve_ips_update_cluster_spec();});

        });

    }


    void SeastarPortMgr::LoadEndpointMapForFile() {
        std::ifstream fin(kEndpointMapFile, std::ios::in);
        if (!fin.good()) {
            LOG(FATAL) << "Load endpoint map file failed.";
        }

        string str;
        while (getline(fin, str)) {
            std::vector<std::string> vec = str_util::Split(str, '=');
            CHECK_EQ(vec.size(), 2);
            endpoint_grpc2seastar_[vec[0]] = vec[1];
        }
    }
}

