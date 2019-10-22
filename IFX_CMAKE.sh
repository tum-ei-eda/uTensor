#!/bin/bash
case `uname` in

  Linux)
	case `lsb_release -is` in
		
		Ubuntu)
			# Ah... the shining highlands of modern Linux desktop Distro sanity
			# ... and Windows-subsystem for Linux
			PREFIX=/opt
		;;
		
		RedHatEnterpriseServer)
			# Oh shit.. looks like the "come back OS/360 all is forgiven" R&D Cluster
			PREFIX=/home/aifordes.work/share
		;;
	
	esac
  ;;
  
  MINGW64*)
	PREFIX=/c/inicio/tools/64
  ;;
  
esac

RV32_TOOOLCHAIN_VERSION=rvtc-ilp32-multlib-1.0.0.0
RV32_TOOLCHAIN_PREFIX=riscv32-unknown-elf
TYPE=Debug
COMPILER=gcc
GCC_HOME="$PREFIX/${RV32_TOOOLCHAIN_VERSION}"
CROSS_SETTING=-DCMAKE_TOOLCHAIN_FILE=ifx_gcc_cross.cmake
CROSS_OPTIONS=( -DCROSS_TOOLS_HOME="$PREFIX/$RV32_TOOOLCHAIN_VERSION" -DCROSS_TOOLS_PREFIX=${RV32_TOOLCHAIN_PREFIX})
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
	-h|--help)
	    echo "Usage: $0 [-h|--help] [-T|--type CmakeBuildType] [-G|--gccbin riscv|host|PathToGccBin] [-X|--xgccprefix GccCrossPrefix]"
		exit 1
	    ;;
	--type|-T)
	    TYPE="$1"
	    shift
	    ;;
	--gcc|-G)
		case "$1" in
		
		riscv)

			;;
		target)
			CROSS_SETTING=
			CROSS_OPTIONS=( )
			;;
		*)
			echo gcc pathname splitting not yet implemented
			exit 1
			;;
		esac
		shift
		;;
	*)
		break;
		;;
    esac
    shift # past argument or value
done


ETISS_HOME="$PREFIX/etiss-0.6" 


mkdir -p "$TYPE"
cmake_opts=( -DCMAKE_BUILD_TYPE="$TYPE" "$CROSS_SETTING" "${CROSS_OPTIONS[@]}" )
(cd "$TYPE"; echo CMAKE_OPTS ${cmake_opts[@]} ; cmake "${cmake_opts[@]}" -G "CodeBlocks - Unix Makefiles" ..)

