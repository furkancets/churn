terraform {
  required_providers {
    # We recommend pinning to the specific version of the Docker Provider you're using
    # since new versions are released frequently
    docker = {
      source  = "kreuzwerker/docker"
      version = "2.23.1"
    }
  }
}
# Configure the docker provider
provider "docker" {
}
# Create a docker image resource
resource "docker_image" "my-fastapi-container" {
  name = "my-fastapi-container"
  build {
    path = "."
    tag  = ["my-fastapi-container:develop"]
    build_arg = {
      name : "my-fastapi-container"
    }
    label = {
      author : "furkan"
    }
  }
}
# Create a docker container resource
resource "docker_container" "fastapi" {
  name    = "fastapi"
  image   = docker_image.my-fastapi-container.image_id
  ports {
    external = 8003
    internal = 8000
  }
}
