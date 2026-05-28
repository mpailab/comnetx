DOCKER_CONTAINER_NAMES=(
    "dev_bokov"
    "dev_uporova"
    "dev_konovalov"
    "dev_egorov"
    "dev_egorov2"
    "dev_drobyshev"
    "dev_drobyshev2"
    "dev_drobyshev3"
)


for c in "${DOCKER_CONTAINER_NAMES[@]}"; do
    echo -en "$c\t: "
    docker exec "$c" bash -lc "(pgrep -afo "$1" || true) | grep -v 'true' | wc -l"
done
