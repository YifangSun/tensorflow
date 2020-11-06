"""Loads a lightweight subset of the ICU library for Unicode processing."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "icu",
        strip_prefix = "icu-release-62-1",
        sha256 = "86b85fbf1b251d7a658de86ce5a0c8f34151027cc60b01e1b76f167379acf181",
        urls = [
            "https://mirror.bazel.build/github.com/unicode-org/icu/archive/release-62-1.tar.gz",
            "https://github.com/unicode-org/icu/archive/release-62-1.tar.gz",
        ],
        build_file = "//third_party/icu:BUILD.bazel",
    )
