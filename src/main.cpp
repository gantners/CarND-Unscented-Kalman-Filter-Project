#include <uWS/uWS.h>
#include <iostream>
#include <fstream>
#include "json.hpp"
#include <math.h>
#include "ukf.h"
#include "tools.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
    auto found_null = s.find("null");
    auto b1 = s.find_first_of("[");
    auto b2 = s.find_first_of("]");
    if (found_null != std::string::npos) {
        return "";
    } else if (b1 != std::string::npos && b2 != std::string::npos) {
        return s.substr(b1, b2 - b1 + 1);
    }
    return "";
}

int main() {
    uWS::Hub h;

    // Create a Kalman Filter instance
    UKF ukf;

    // used to compute the RMSE later
    Tools tools;
    vector<VectorXd> estimations;
    vector<VectorXd> ground_truth;
    static string separator = "\t";
    static int timeStep = 1;
    static string filePathStr = "../output/NIS.txt";
    static const char* filePath = filePathStr.c_str();
    ofstream NIS_file_;

    h.onMessage([&ukf, &tools, &estimations, &ground_truth, &NIS_file_](uWS::WebSocket <uWS::SERVER> ws, char *data,
                                                                        size_t length,
                                                                        uWS::OpCode opCode) {
                    // "42" at the start of the message means there's a websocket message event.
                    // The 4 signifies a websocket message
                    // The 2 signifies a websocket event

                    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

                        auto s = hasData(std::string(data));
                        if (s != "") {

                            auto j = json::parse(s);

                            std::string event = j[0].get<std::string>();

                            if (event == "telemetry") {
                                // j[1] is the data JSON object

                                //Those values have been evaluated from a series of measurements with

                                ukf.std_a_ = 1.5;
                                ukf.std_yawdd_ = 0.5;

                                string sensor_measurment = j[1]["sensor_measurement"];

                                MeasurementPackage meas_package;
                                istringstream iss(sensor_measurment);
                                long long timestamp;

                                // reads first element from the current line
                                string sensor_type;
                                iss >> sensor_type;

                                if (sensor_type.compare("L") == 0) {
                                    meas_package.sensor_type_ = MeasurementPackage::LASER;
                                    meas_package.raw_measurements_ = VectorXd(2);
                                    float px;
                                    float py;
                                    iss >> px;
                                    iss >> py;
                                    meas_package.raw_measurements_ << px, py;
                                    iss >> timestamp;
                                    meas_package.timestamp_ = timestamp;
                                } else if (sensor_type.compare("R") == 0) {

                                    meas_package.sensor_type_ = MeasurementPackage::RADAR;
                                    meas_package.raw_measurements_ = VectorXd(3);
                                    float ro;
                                    float theta;
                                    float ro_dot;
                                    iss >> ro;
                                    iss >> theta;
                                    iss >> ro_dot;
                                    meas_package.raw_measurements_ << ro, theta, ro_dot;
                                    iss >> timestamp;
                                    meas_package.timestamp_ = timestamp;
                                }
                                float x_gt;
                                float y_gt;
                                float vx_gt;
                                float vy_gt;
                                iss >> x_gt;
                                iss >> y_gt;
                                iss >> vx_gt;
                                iss >> vy_gt;
                                VectorXd gt_values(4);
                                gt_values(0) = x_gt;
                                gt_values(1) = y_gt;
                                gt_values(2) = vx_gt;
                                gt_values(3) = vy_gt;
                                ground_truth.push_back(gt_values);

                                //Call ProcessMeasurment(meas_package) for Kalman filter
                                ukf.ProcessMeasurement(meas_package);

                                //Push the current estimated x,y positon from the Kalman filter's state vector

                                VectorXd estimate(4);

                                double p_x = ukf.x_(0);
                                double p_y = ukf.x_(1);
                                double v = ukf.x_(2);
                                double yaw = ukf.x_(3);

                                double v1 = cos(yaw) * v;
                                double v2 = sin(yaw) * v;

                                estimate(0) = p_x;
                                estimate(1) = p_y;
                                estimate(2) = v1;
                                estimate(3) = v2;

                                estimations.push_back(estimate);

                                VectorXd RMSE = tools.CalculateRMSE(estimations, ground_truth);

                                // output the NIS values
                                if (timeStep++ % 498 == 0) {
                                    NIS_file_.open(filePath, std::ios::app);

                                    NIS_file_ << (ukf.previous_timestamp_ - ukf.initial_timestamp_) / 1000 << separator;

                                    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
                                        NIS_file_ << "L" << separator;
                                    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
                                        NIS_file_ << "R" << separator;
                                    } else {
                                        NIS_file_ << "-" << separator;
                                    }

                                    // Process noise standard deviation longitudinal acceleration in m/s^2
                                    NIS_file_ << ukf.std_a_ << separator;
                                    // Process noise standard deviation yaw acceleration in rad/s^2
                                    NIS_file_ << ukf.std_yawdd_ << separator;
                                    // Laser measurement noise standard deviation position1 in m
                                    NIS_file_ << ukf.std_laspx_ << separator;
                                    // Laser measurement noise standard deviation position2 in m
                                    NIS_file_ << ukf.std_laspy_ << separator;
                                    // Radar measurement noise standard deviation radius in m
                                    NIS_file_ << ukf.std_radr_ << separator;
                                    // Radar measurement noise standard deviation angle in rad
                                    NIS_file_ << ukf.std_radphi_ << separator;
                                    // Radar measurement noise standard deviation radius change in m/s
                                    NIS_file_ << ukf.std_radrd_ << separator;

                                    NIS_file_ << ukf.x_(0) << separator;
                                    NIS_file_ << ukf.x_(1) << separator;
                                    NIS_file_ << ukf.x_(2) << separator;
                                    NIS_file_ << ukf.x_(3) << separator;
                                    NIS_file_ << ukf.x_(4) << separator;

                                    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
                                        NIS_file_ << meas_package.raw_measurements_(0) << separator;
                                        NIS_file_ << meas_package.raw_measurements_(1) << separator;
                                        NIS_file_ << to_string(ukf.NIS_lidar_) << separator;
                                    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
                                        NIS_file_ << meas_package.raw_measurements_(0) * cos(meas_package.raw_measurements_(1))
                                                  << separator;
                                        NIS_file_ << meas_package.raw_measurements_(0) * sin(meas_package.raw_measurements_(1))
                                                  << separator;
                                        NIS_file_ << to_string(ukf.NIS_radar_) << separator;
                                    } else {
                                        NIS_file_ << "-" << separator;
                                        NIS_file_ << "-" << separator;
                                        NIS_file_ << "-" << separator;
                                    }

                                    NIS_file_ << gt_values(0) << separator;
                                    NIS_file_ << gt_values(1) << separator;
                                    NIS_file_ << gt_values(2) << separator;
                                    NIS_file_ << gt_values(3) << separator;

                                    NIS_file_ << RMSE(0) << separator;
                                    NIS_file_ << RMSE(1) << separator;
                                    NIS_file_ << RMSE(2) << separator;
                                    NIS_file_ << RMSE(3) << separator;

                                    NIS_file_ << "\r\n";
                                    NIS_file_.close();
                                }

                                json msgJson;
                                msgJson["estimate_x"] = p_x;
                                msgJson["estimate_y"] = p_y;
                                msgJson["rmse_x"] = RMSE(0);
                                msgJson["rmse_y"] = RMSE(1);
                                msgJson["rmse_vx"] = RMSE(2);
                                msgJson["rmse_vy"] = RMSE(3);
                                auto msg = "42[\"estimate_marker\"," + msgJson.dump() + "]";
                                // std::cout << msg << std::endl;
                                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);

                            }
                        } else {

                            std::string msg = "42[\"manual\",{}]";
                            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
                        }
                    }

                }

    );

// We don't need this since we're not using HTTP but if it's removed the program
// doesn't compile :-(
    h.onHttpRequest([](
            uWS::HttpResponse *res, uWS::HttpRequest
    req,
            char *data, size_t, size_t
    ) {
        const std::string s = "<h1>Hello world!</h1>";
        if (req.

                        getUrl()

                    .valueLength == 1) {
            res->
                    end(s
                                .

                                        data(), s

                                .

                                        length()

            );
        } else {
// i guess this should be done more gracefully?
            res->end(nullptr, 0);
        }
    });

    h.onConnection([&h](
            uWS::WebSocket <uWS::SERVER> ws, uWS::HttpRequest
    req) {
        std::cout << "Connected!!!" <<
                  std::endl;
    });

    h.onDisconnection([&h](
            uWS::WebSocket <uWS::SERVER> ws,
            int code,
            char *message, size_t
            length) {
        ws.

                close();

        std::cout << "Disconnected" <<
                  std::endl;
    });

    if (false) {

//Delete old file
        remove(filePath);
// Create new file
        NIS_file_.open(filePath, std::ios::app);

//print header row
        NIS_file_ << "timestamp" <<
                  separator;

        NIS_file_ << "sensor_type" <<
                  separator;
// Process noise standard deviation longitudinal acceleration in m/s^2
        NIS_file_ << "std_a_" <<
                  separator;
// Process noise standard deviation yaw acceleration in rad/s^2
        NIS_file_ << "std_yawdd_" <<
                  separator;
// Laser measurement noise standard deviation position1 in m
        NIS_file_ << "std_laspx_" <<
                  separator;
// Laser measurement noise standard deviation position2 in m
        NIS_file_ << "std_laspy_" <<
                  separator;
// Radar measurement noise standard deviation radius in m
        NIS_file_ << "std_radr_" <<
                  separator;
// Radar measurement noise standard deviation angle in rad
        NIS_file_ << "std_radphi_" <<
                  separator;
// Radar measurement noise standard deviation radius change in m/s
        NIS_file_ << "std_radrd_" <<
                  separator;

        NIS_file_ << "Estimated px" <<
                  separator;
        NIS_file_ << "Estimated py" <<
                  separator;
        NIS_file_ << "Estimated vx" <<
                  separator;
        NIS_file_ << "Estimated yaw_angle" <<
                  separator;
        NIS_file_ << "Estimated yaw_rate" <<
                  separator;

        NIS_file_ << "Measured px" <<
                  separator;
        NIS_file_ << "Measured py" <<
                  separator;
        NIS_file_ << "NIS" <<
                  separator;

        NIS_file_ << "GT px" <<
                  separator;
        NIS_file_ << "GT py" <<
                  separator;
        NIS_file_ << "GT vx" <<
                  separator;
        NIS_file_ << "GT vy" <<
                  separator;

        NIS_file_ << "RSME px" <<
                  separator;
        NIS_file_ << "RSME py" <<
                  separator;
        NIS_file_ << "RSME vx" <<
                  separator;
        NIS_file_ << "RSME vy" <<
                  separator;

        NIS_file_ << "\r\n";
        NIS_file_.

                close();
    }
    int port = 4567;
    if (h.
            listen(port)
            ) {
        std::cout << "Listening to port " << port <<
                  std::endl;
    } else {
        std::cerr << "Failed to listen to port" <<
                  std::endl;
        return -1;
    }
    h.

            run();

}























































































