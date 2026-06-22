#!/usr/bin/env bash

set -ex

if [ ! -n "$1" ];then
    echo "colmap_to_mvs.sh [imgdir] [colmap result folder, containing cameras.txt points3D.txt]"
    exit 1
fi

imgdir=$(realpath $1)
colmap_result=$(realpath $2)

ws=`pwd`/mvs_output
if [ ! -d $ws ];then
    mkdir -p $ws
fi
cd $ws

root=/usr/local/bin/OpenMVS

function interface_colmap() {
    ln -s $colmap_result $ws/sparse
    ln -s $imgdir $ws/images
    $root/InterfaceCOLMAP \
        --working-folder $ws \
        --input-file $ws \
        --output-file $ws/model_colmap.mvs
}
interface_colmap

function densify_pointcloud() {
    $root/DensifyPointCloud \
        --input-file $ws/model_colmap.mvs \
        --working-folder $ws \
        --output-file $ws/model_dense.mvs \
        --archive-type -1
}
densify_pointcloud

function reconstruct_mesh() {
    $root/ReconstructMesh \
        --working-folder $ws/ \
        --input-file $ws/model_dense.mvs \
        --output-file $ws/model_dense_mesh.mvs
}
reconstruct_mesh

function refine_mesh() {
    $root/RefineMesh \
        --working-folder $ws \
        --resolution-level 1 \
        --cuda-device -1 \
        --input-file $ws/model_dense_mesh.mvs \
        --output-file $ws/model_dense_mesh_refine.mvs
}
refine_mesh

function texture_mesh() {
    $root/TextureMesh \
        --working-folder $ws \
        --export-type obj \
        --cuda-device -1 \
        --input-file $ws/model_dense_mesh_refine.mvs \
        --output-file $ws/model.obj
}
texture_mesh
