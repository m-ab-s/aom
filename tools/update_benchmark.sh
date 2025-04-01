#!/bin/bash

# Update third_party/benchmark to the latest version.

# Usage: (under libaom root directory)
# ./tools/update_benchmark.sh

set -e

benchmark_dir="$(pwd)/third_party/benchmark"
repo_url="https://github.com/google/benchmark"

git clone --depth 1 "$repo_url" "$benchmark_dir"

cd "${benchmark_dir}"

commit_hash=$(git rev-parse HEAD)

# Remove everything except ./src and ./include
find . -mindepth 1 \
  -not -path "./src" \
  -not -path "./src/*" \
  -not -path "./include" \
  -not -path "./include/*" \
  -not -name "LICENSE" \
  -delete

# Remove markdown files
find . -name "*.md" -delete

# Update the include path
find ./src \( -name "*.c" -o -name "*.cc" -o -name "*.h" \) -print0 | \
  xargs -0 sed -i \
  's/#include "benchmark\//#include "third_party\/benchmark\/include\/benchmark\//g'

find ./include \( -name "*.c" -o -name "*.cc" -o -name "*.h" \) -print0 | \
  xargs -0 sed -i \
  's/#include "benchmark\//#include "third_party\/benchmark\/include\/benchmark\//g'

find ./src \( -name "*.c" -o -name "*.cc" -o -name "*.h" \) -print0 | \
  xargs -0 sed -i \
  's/#include <benchmark\//#include <third_party\/benchmark\/include\/benchmark\//g'

cat > "${benchmark_dir}/README.libaom" <<EOF
URL: $repo_url

Version: $commit_hash
License: Apache License 2.0
License File: LICENSE

Description:
Benchmark is a library to benchmark code snippets, similar to unit tests.

Local Changes:
Remove everything except src/ include/ and LICENSE
EOF
