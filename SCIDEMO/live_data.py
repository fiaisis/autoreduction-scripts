from mantid.simpleapi import SaveNexusProcessed

print("hello world")
SaveNexusProcessed(
    InputWorkspace="live-ws",
    Filename="/Users/sham/work/plotting-service/plotting-service/test/test_ceph/MARI/RBNumber/RB20024/autoreduced/foo.nxs",
)
