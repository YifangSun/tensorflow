From cr.d.xiaomi.net/feeds/centos7_apollo_gcc8:0.3
RUN yum install -y libunwind.x86_64 libunwind-devel.x86_64 patch libaio.x86_64 libaio-devel.x86_64 lksctp-tools-1.0.17-2.el7.x86_64 lksctp-tools-devel.x86_64 xfsprogs.x86_64 xfsprogs-devel.x86_64 systemtap-sdt-devel numactl-libs.x86_64 numad.x86_64 numactl-devel.x86_64 python-enum34
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py
RUN pip install -y numpy enum keras_applications keras_preprocessing
