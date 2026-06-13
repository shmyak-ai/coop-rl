#!/usr/bin/env bash
# Regenerate mlagents_envs protobuf/grpc stubs for this project's protobuf 6.x.
#
# mlagents-envs ships stubs built for protobuf<3.21, but coop-rl pins protobuf 6.x
# (required by ray/tensorflow/tensorboard/orbax), so a fresh install fails to
# `import mlagents_envs`. This regenerates the stubs from the .proto sources using
# grpcio-tools matching the installed grpcio. The output lives in the venv, so it
# is lost on `uv sync`/reinstall of mlagents-envs — re-run this script afterwards.
#
# Usage: scripts/regen_mlagents_protos.sh
#   MLAGENTS_PROTO_DIR overrides the .proto source directory
#   (default: $HOME/src/ml-agents/protobuf-definitions/proto).
set -euo pipefail

PROTO_DIR="${MLAGENTS_PROTO_DIR:-$HOME/src/ml-agents/protobuf-definitions/proto}"
PKG_PROTO="$PROTO_DIR/mlagents_envs/communicator_objects"

if [ ! -d "$PKG_PROTO" ]; then
    echo "proto sources not found at $PKG_PROTO" >&2
    echo "set MLAGENTS_PROTO_DIR to the ml-agents 'protobuf-definitions/proto' directory" >&2
    exit 1
fi

# site-packages root that contains the installed mlagents_envs package.
SP_ROOT="$(uv run python -c 'import mlagents_envs, os; print(os.path.dirname(os.path.dirname(mlagents_envs.__file__)))')"
# grpc generated code carries a guard requiring an exact-or-newer grpcio; match it.
GRPCIO_VERSION="$(uv run python -c 'import grpc; print(grpc.__version__)')"

echo "proto sources : $PROTO_DIR"
echo "output root   : $SP_ROOT"
echo "grpcio-tools  : ==$GRPCIO_VERSION (matching installed grpcio)"

# Messages: *_pb2.py for every proto. Service: *_pb2_grpc.py for unity_to_external.
uv run --with "grpcio-tools==$GRPCIO_VERSION" python -m grpc_tools.protoc \
    -I"$PROTO_DIR" --python_out="$SP_ROOT" \
    "$PKG_PROTO"/*.proto
uv run --with "grpcio-tools==$GRPCIO_VERSION" python -m grpc_tools.protoc \
    -I"$PROTO_DIR" --python_out="$SP_ROOT" --grpc_python_out="$SP_ROOT" \
    "$PKG_PROTO"/unity_to_external.proto

uv run python -c "from mlagents_envs.environment import UnityEnvironment; print('mlagents_envs import OK')"
