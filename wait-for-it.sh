#!/usr/bin/env bash
#   Use this script to test if a given TCP host/port are available

set -e

TIMEOUT=15
QUIET=0
STRICT=0
HOST=
PORT=
WAITFORIT_cmd=''

echoerr() {
    if [ "$QUIET" -ne 1 ]; then echo "$@" 1>&2; fi
}

usage() {
    exitcode="$1"
    cat << USAGE >&2
Usage:
    wait-for-it.sh host:port [-s] [-t timeout] [-- command args]
    -h HOST | --host=HOST       Host or IP under test
    -p PORT | --port=PORT       TCP port under test
                                Alternatively, you specify the host and port as host:port
    -s | --strict               Only execute subcommand if the test succeeds
    -q | --quiet                Don't output any status messages
    -t TIMEOUT | --timeout=TIMEOUT
                                Timeout in seconds, zero for no timeout
    -- COMMAND ARGS             Execute command with args after the test finishes
USAGE
    exit "$exitcode"
}

wait_for() {
    for i in $(seq $TIMEOUT); do
        nc -z "$HOST" "$PORT" > /dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        *:* )
        HOST=$(printf "%s\n" "$1"| cut -d : -f 1)
        PORT=$(printf "%s\n" "$1"| cut -d : -f 2)
        shift 1
        ;;
        -h)
        HOST="$2"
        if [ "$HOST" == "" ]; then break; fi
        shift 2
        ;;
        --host=*)
        HOST=$(printf "%s" "$1" | cut -d = -f 2)
        shift 1
        ;;
        -p)
        PORT="$2"
        if [ "$PORT" == "" ]; then break; fi
        shift 2
        ;;
        --port=*)
        PORT=$(printf "%s" "$1" | cut -d = -f 2)
        shift 1
        ;;
        -q | --quiet)
        QUIET=1
        shift 1
        ;;
        -s | --strict)
        STRICT=1
        shift 1
        ;;
        -t)
        TIMEOUT="$2"
        if [ "$TIMEOUT" == "" ]; then break; fi
        shift 2
        ;;
        --timeout=*)
        TIMEOUT=$(printf "%s" "$1" | cut -d = -f 2)
        shift 1
        ;;
        --)
        shift
        WAITFORIT_cmd="$@"
        break
        ;;
        --help)
        usage 0
        ;;
        *)
        echoerr "Unknown argument: $1"
        usage 1
        ;;
    esac
done

if [ "$HOST" == "" ] || [ "$PORT" == "" ]; then
    echoerr "Error: you need to provide a host and port to test."
    usage 2
fi

wait_for

if [ $? -ne 0 ]; then
    echo "wait-for-it.sh: timeout occurred after waiting $TIMEOUT seconds for $HOST:$PORT"
    if [ "$STRICT" -eq 1 ]; then exit 1; fi
fi

if [ "$WAITFORIT_cmd" != "" ]; then
    exec $WAITFORIT_cmd
else
    exit 0
fi
