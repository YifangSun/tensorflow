## build_image
- cr.d.xiaomi.net/deploy/centos7_apollo_gcc8_tf:0.1

## 说明
- gcc8,bazel-0.20.0
- base image:cr.d.xiaomi.net/feeds/centos7_apollo_gcc8:0.3
- install lib:
-- yum:  libunwind.x86_64,libunwind-devel.x86_64,patch,libaio.x86_64,libaio-devel.x86_64,lksctp-tools-1.0.17-2.el7.x86_64,lksctp-tools-devel.x86_64,xfsprogs.x86_64,xfsprogs-devel.x86_64,systemtap-sdt-devel,numactl-libs.x86_64,numad.x86_64,numactl-devel.x86_64,python-enum34
-- pip:	numpy,enum,keras_applications,keras_preprocessing
