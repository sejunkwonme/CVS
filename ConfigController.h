#pragma once

#include <QObject>


class ConfigController : public QObject {
	Q_OBJECT

public:
	ConfigController(QObject *parent);
	~ConfigController();

    std::string path_;
    

    struct AppConfig {
        std::string name;
        std::string version;
        std::string log_level;
        std::string save_path;
    };

    struct CameraConfig {
        int id;
        std::string backend;
        int width;
        int height;
        float fps;
        bool auto_exposure;
        int exposure;
    };

    struct ModelConfig {
        std::string name;
        std::string path;
        std::vector<std::string> providers;
        int input_width;
        int input_height;
        float conf_th;
        float nms_th;
    };

    struct UIConfig {
        std::string theme;
        bool draw_boxes;
        bool show_fps;
        bool show_time;
    };

    struct DatabaseConfig {
        std::string type;
        std::string path;
    };

    struct Config {
        AppConfig app;
        CameraConfig camera;
        ModelConfig model;
        UIConfig ui;
        DatabaseConfig database;
    };

    Config conf_;

    void loadConfig(const std::string& path);

    void initialize();
	
signals:
    void initialized();
};