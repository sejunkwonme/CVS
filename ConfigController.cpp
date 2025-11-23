#include "ConfigController.h"
#include <yaml-cpp/yaml.h>

ConfigController::ConfigController(QObject *parent)
: QObject(parent),
path_("D:/Repo/CVS/setting.yml"){
	
}

ConfigController::~ConfigController() {
	
}

void ConfigController::loadConfig(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);

    // App
    conf_.app.name = root["app"]["name"].as<std::string>();
    conf_.app.version = root["app"]["version"].as<std::string>();
    conf_.app.log_level = root["app"]["log_level"].as<std::string>();
    conf_.app.save_path = root["app"]["save_path"].as<std::string>();

    // Camera
    conf_.camera.id = root["camera"]["id"].as<int>();
    conf_.camera.backend = root["camera"]["backend"].as<std::string>();
    conf_.camera.width = root["camera"]["resolution"]["width"].as<int>();
    conf_.camera.height = root["camera"]["resolution"]["height"].as<int>();
    conf_.camera.fps = root["camera"]["fps"].as<float>();
    conf_.camera.auto_exposure = root["camera"]["auto_exposure"].as<bool>();
    conf_.camera.exposure = root["camera"]["exposure"].as<int>();

    // Model
    conf_.model.name = root["model"]["name"].as<std::string>();
    conf_.model.path = root["model"]["path"].as<std::string>();
    conf_.model.providers = root["model"]["providers"].as<std::vector<std::string>>();
    conf_.model.input_width = root["model"]["input"]["width"].as<int>();
    conf_.model.input_height = root["model"]["input"]["height"].as<int>();
    conf_.model.conf_th = root["model"]["threshold"]["conf"].as<float>();
    conf_.model.nms_th = root["model"]["threshold"]["nms"].as<float>();

    // UI
    conf_.ui.theme = root["ui"]["theme"].as<std::string>();
    conf_.ui.draw_boxes = root["ui"]["camera_view"]["draw_boxes"].as<bool>();
    conf_.ui.show_fps = root["ui"]["camera_view"]["show_fps"].as<bool>();
    conf_.ui.show_time = root["ui"]["camera_view"]["show_inference_time"].as<bool>();

    // Database
    conf_.database.type = root["database"]["type"].as<std::string>();
    conf_.database.path = root["database"]["path"].as<std::string>();
}

void ConfigController::initialize() {
    loadConfig(path_);
    emit initialized();
}