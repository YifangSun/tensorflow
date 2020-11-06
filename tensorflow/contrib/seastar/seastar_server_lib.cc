#include "tensorflow/contrib/seastar/seastar_server_lib.h"

#include <fstream>
#include <unordered_map>

#include "grpc/support/alloc.h"
#include "tensorflow/contrib/seastar/seastar_rendezvous_mgr.h"
#include "tensorflow/contrib/seastar/seastar_worker_cache.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mem.h"


namespace tensorflow {

SeastarServer::SeastarServer(const ServerDef& server_def, Env* env)
    : GrpcServer(server_def, env) {
  seastar_port_mgr_ = std::shared_ptr<SeastarPortMgr>(new SeastarPortMgr(server_def));
}

SeastarServer::~SeastarServer() {
  delete seastar_worker_service_;
  delete seastar_engine_;
}

Status SeastarServer::Init() {
  seastar_worker_impl_ = NewSeastarWorker(worker_env());
  seastar_worker_service_ =
      NewSeastarWorkerService(seastar_worker_impl_.get()).release();
  seastar_bound_port_ = seastar_port_mgr_->GetLocalSeastarPort();
  seastar_engine_ =
      new SeastarEngine(seastar_bound_port_, seastar_worker_service_, seastar_port_mgr_);

  RendezvousMgrCreationFunction rendezvous_mgr_func =
      [this](const WorkerEnv* env) { return new SeastarRendezvousMgr(env); };

  GrpcServerOptions opts;
  opts.rendezvous_mgr_func = rendezvous_mgr_func;
  return GrpcServer::Init(opts);
}

Status SeastarServer::ParseChannelSpec(const WorkerCacheFactoryOptions& options,
                                       SeastarChannelSpec* channel_spec) {
  for (const auto& job : options.cluster_def->job()) {
    std::unordered_map<int, string> host_ports;
    for (const auto& task : job.tasks()) {
      string& host_port = host_ports[task.first];
      if (!host_port.empty()) {
        return errors::InvalidArgument("JobDef for job \"", job.name(),
                                       "\" specified two addresses for task \"",
                                       task.first, "\": ", host_port, " and ",
                                       task.second);
      }
      if (job.name() == *options.job_name && task.first == options.task_index) {
        host_port = strings::StrCat("localhost:", seastar_bound_port_);
      } else {
        host_port = task.second;
        int grpc_port = -1;
        const std::vector<string> vec = str_util::Split(host_port, ':');
        if (vec.size() != 2 || !strings::safe_strto32(vec[1], &grpc_port)) {
          LOG(ERROR) << "error host port schema " << host_port;
          return errors::Cancelled("error host port schema ", host_port);
        }

        std::string seastar_host_port =
            seastar_port_mgr_->GetSeastarIpPort(job.name(), task.first);
        LOG(INFO) << "host port: " << host_port
                  << ", remote seastar host port: " << seastar_host_port;
        host_port = seastar_host_port;
      }
    }

    TF_RETURN_IF_ERROR(channel_spec->AddHostPortsJob(job.name(), host_ports));
  }
  return Status::OK();
}

Status SeastarServer::WorkerCacheFactory(
    const WorkerCacheFactoryOptions& options,
    WorkerCacheInterface** worker_cache) {
  if (options.job_name == nullptr || options.job_name->empty()) {
    Status s = errors::InvalidArgument(
        "The master (current machine) is not included in the provided "
        "cluster_def. ",
        options.cluster_def->DebugString());
    LOG(WARNING) << s;
    return s;
  }

  SeastarChannelSpec channel_spec;
  TF_RETURN_IF_ERROR(ParseChannelSpec(options, &channel_spec));
  std::unique_ptr<SeastarChannelCache> channel_cache(
      NewSeastarChannelCache(seastar_engine_, channel_spec));

  string name_prefix = strings::StrCat("/job:", *options.job_name, "/replica:0",
                                       "/task:", options.task_index);

  const string host_port = channel_cache->TranslateTask(name_prefix);
  int requested_port;

  if (!strings::safe_strto32(str_util::Split(host_port, ':')[1],
                             &requested_port)) {
    return errors::Internal("Could not parse port for local server from \"",
                            channel_cache->TranslateTask(name_prefix), "\".");
  }

  LOG(INFO) << "SeastarWorkerCacheFactory, name_prefix:" << name_prefix;
  *worker_cache = NewSeastarWorkerCacheWithLocalWorker(
      channel_cache.release(), seastar_worker_impl_.get(), name_prefix,
      worker_env());

  return Status::OK();
}

Status SeastarServer::Create(const ServerDef& server_def, Env* env,
                             std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<SeastarServer> ret(
      new SeastarServer(server_def, env == nullptr ? Env::Default() : env));
  Status s = ret->Init();
  if (!s.ok()) {
    LOG(ERROR) << s;
    return s;
  }
  *out_server = std::move(ret);
  return Status::OK();
}

namespace {

class SeastarServerFactory : public ServerFactory {
 public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return server_def.protocol() == "grpc+seastar";
  }

  Status NewServer(const ServerDef& server_def,
                   std::unique_ptr<ServerInterface>* out_server) override {
    return SeastarServer::Create(server_def, Env::Default(), out_server);
  }
};

class SeastarServerRegistrar {
 public:
  SeastarServerRegistrar() {
    gpr_allocation_functions alloc_fns;
    memset(&alloc_fns, 0, sizeof(alloc_fns));
    alloc_fns.malloc_fn = port::Malloc;
    alloc_fns.realloc_fn = port::Realloc;
    alloc_fns.free_fn = port::Free;
    gpr_set_allocation_functions(alloc_fns);
    ServerFactory::Register("SEASTAR_SERVER", new SeastarServerFactory());
  }
};

static SeastarServerRegistrar registrar;

}  // namespace

}  // namespace tensorflow
