#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_lidar_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 0.2;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.2;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03; // 0.0175

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3; //0.1

    /*
     * UKF Init
     * */

    ///* initially set to false, set to true in first call of ProcessMeasurement
    is_initialized_ = false;

    ///* State dimension
    n_x_ = 5;

    initial_timestamp_ = 0;

    /**
     * You won't know where the bicycle is until you receive the first sensor measurement. Once the first sensor measurement
     * arrives, you can initialize p​x and p​y.
     * For the other variables in the state vector x, you can try different initialization values to see what works best
     */
    // initial state vector
    x_ = VectorXd::Zero(n_x_);

    /**
     * Why is the identity matrix a good place to start? Since the non-diagonal values represent
     * the covariances between variables, the P matrix is symmetrical. The identity matrix is also symmetrical.
     * The symmetry comes from the fact that the covariance between two variables is the same
     * whether you look at (x, y) or (y, x): σ ​px,py = σ py,px
     * If you print out the P matrix in your UKF code, you will see that P remains symmetrical even after the
     * matrix gets updated. If your solution is working, you will also see that P starts to converge to small
     * values relatively quickly.
     * Instead of setting each of the diagonal values to 1, you can try setting the diagonal values by
     * how much difference you expect between the true state and the initialized x state vector.
     * For example, in the project, we assume the standard deviation of the lidar x and y measurements is 0.15.
     * If we initialized px with a lidar measurement, the initial variance or uncertainty in px would probably be less than 1.
     */
    // initial covariance matrix along identity line
    P_ = MatrixXd::Zero(n_x_, n_x_);
    for (int i = 0; i < n_x_; ++i) {
        P_(i, i) = 0.15;
    }

    ///* Augmented state dimension
    n_aug_ = 7;

    ///* Sigma point spreading parameter
    lambda_ = 3 - n_x_;

    // Number of sigma points
    n_sigma = 2 * n_aug_ + 1;

    Xsig_pred_ = MatrixXd(n_sigma, n_x_);

    //Number of radar
    n_z_radar = 3;

    //Number of lidar
    n_z_lidar = 2;

    NIS_lidar_ = 0.0;

    NIS_radar_ = 0.0;

    ///* Weights of sigma points
    weights_ = VectorXd(2 * n_aug_ + 1);
    double weight_0 = lambda_ / (lambda_ + n_aug_);
    weights_(0) = weight_0;
    for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
        double weight = 0.5 / (n_aug_ + lambda_);
        weights_(i) = weight;
    }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    //Check if a sensor is disabled and return right away avoiding unnecessary calculations
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        if (use_radar_){
            //cout << "Incoming radar measurement" << endl;
        }
        else{
            //cout << "Radar disabled" << endl;
            return;
        }
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        if (use_lidar_){
            //cout << "Incoming lidar measurement" << endl;
        }
        else{
            //cout << "Lidar disabled" << endl;
            return;
        }
    }

    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        //cout << "Init" << endl;
        Init(meas_package);
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/
    //cout << "Prediction time" << endl;
    //compute the time elapsed between the current and previous measurements
    double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;

    if (dt <= 0.0001) {
        //cout << "Discarding second measure within short timeframe" << endl;
        return;
    }
    previous_timestamp_ = meas_package.timestamp_;

    //cout << "Prediction" << endl;

    //Generate sigma points
    MatrixXd Xsig = MatrixXd(2 * n_x_ + 1, n_x_);
    GenerateSigmaPoints(&Xsig);

    //Augment sigma points
    MatrixXd Xsig_aug = MatrixXd(n_sigma, n_aug_);
    AugmentedSigmaPoints(&Xsig_aug);

    VectorXd x_pred = VectorXd(n_x_);
    MatrixXd P_pred = MatrixXd(n_x_, n_x_);
    Prediction(dt, &x_pred, &P_pred, &Xsig_pred_, Xsig_aug);

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        //cout << "Update Radar" << endl;
        UpdateRadar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        //cout << "Update Lidar" << endl;
        UpdateLidar(meas_package);
    }

    // print the output
    //cout << "x_: " << ekf_.x_ << endl;
    //cout << "P_: " << ekf_.P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t,
                     VectorXd *x_pred,
                     MatrixXd *P_pred, MatrixXd *Xsig_pred, MatrixXd &Xsig_aug) {
    /**
    Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */

    //Predict sigma points
    SigmaPointPrediction(Xsig_pred, Xsig_aug, delta_t);

    //Predict mean and covariance
    PredictMeanAndCovariance(x_pred, P_pred, *Xsig_pred);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the lidar NIS.
    */

    VectorXd z = meas_package.raw_measurements_;

    //Predict measurement
    VectorXd z_pred = VectorXd(3);
    MatrixXd S_pred = MatrixXd(3, 3);

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z_lidar, 2 * n_aug_ + 1);
    PredictLidarMeasurement(&z_pred, &S_pred, Zsig, Xsig_pred_);
    //std::cout << "z_pred = " << std::endl << z_pred << std::endl;
    //std::cout << "S_pred = " << std::endl << S_pred << std::endl;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z_lidar);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S_pred.inverse();

    //residual
    VectorXd z_diff = z - z_pred;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S_pred * K.transpose();

    //print result
    //std::cout << "Updated state x: " << std::endl << x_ << std::endl;
    //std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;

    //write result
    //*x_out = x_;
    //*P_out = P_;

    NIS_lidar_= z_diff.transpose() * S_pred.inverse() * z_diff;
}

void UKF::PredictLidarMeasurement(VectorXd *z_out, MatrixXd *S_out, MatrixXd &Zsig, MatrixXd &Xsig_pred){
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        // extract values for better readibility
        double p_x = Xsig_pred(0, i);
        double p_y = Xsig_pred(1, i);

        // measurement model
        Zsig(0, i) = p_x;
        Zsig(1, i) = p_y;
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z_lidar);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z_lidar, n_z_lidar);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z_lidar, n_z_lidar);
    R << std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;
    S = S + R;

    //write result
    *z_out = z_pred;
    *S_out = S;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the radar NIS.
    */

    VectorXd z = meas_package.raw_measurements_;

    //Predict measurement
    VectorXd z_pred = VectorXd(3);
    MatrixXd S_pred = MatrixXd(3, 3);

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z_radar, 2 * n_aug_ + 1);
    PredictRadarMeasurement(&z_pred, &S_pred, Zsig, Xsig_pred_);
    //std::cout << "z_pred = " << std::endl << z_pred << std::endl;
    //std::cout << "S_pred = " << std::endl << S_pred << std::endl;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z_radar);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S_pred.inverse();

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S_pred * K.transpose();

    //print result
    //std::cout << "Updated state x: " << std::endl << x_ << std::endl;
    //std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;

    //write result
    //*x_out = x_;
    //*P_out = P_;

    NIS_radar_ = z_diff.transpose() * S_pred.inverse() * z_diff;
}

void UKF::GenerateSigmaPoints(MatrixXd *Xsig_out) {
    //cout << "GenerateSigmaPoints" << endl;
    //create sigma point matrix
    MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);
    //calculate square root of P
    MatrixXd A = P_.llt().matrixL();
    //set first column of sigma point matrix
    Xsig.col(0) = x_;
    //set remaining sigma points
    for (int i = 0; i < n_x_; i++) {
        Xsig.col(i + 1) = x_ + sqrt(lambda_ + n_x_) * A.col(i);
        Xsig.col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
    }
    //write result
    *Xsig_out = Xsig;
    //std::cout << "Xsig = " << std::endl << Xsig << std::endl;
}

void UKF::AugmentedSigmaPoints(MatrixXd *Xsig_out) {
    //cout << "AugmentedSigmaPoints" << endl;
    //create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented mean state
    x_aug.head(5) = x_.head(5);
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5, 5) = P_;
    P_aug(5, 5) = std_a_ * std_a_;
    P_aug(6, 6) = std_yawdd_ * std_yawdd_;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }

    //write result
    *Xsig_out = Xsig_aug;
    //std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;
}

void UKF::SigmaPointPrediction(MatrixXd *Xsig_out, MatrixXd &Xsig_aug, double delta_t) {
    //cout << "SigmaPointPrediction" << endl;
    //create matrix with predicted sigma points as columns
    MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

    //predict sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //extract values for better readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        } else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + nu_a * delta_t;

        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;

        //write predicted sigma point into right column
        Xsig_pred(0, i) = px_p;
        Xsig_pred(1, i) = py_p;
        Xsig_pred(2, i) = v_p;
        Xsig_pred(3, i) = yaw_p;
        Xsig_pred(4, i) = yawd_p;

        //write result
        *Xsig_out = Xsig_pred;
        //std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;
    }
}

void UKF::PredictMeanAndCovariance(VectorXd *x_pred, MatrixXd *P_pred, MatrixXd &Xsig_pred) {
    //cout << "PredictMeanAndCovariance" << endl;
    //create vector for weights
    VectorXd weights = weights_;//VectorXd(2 * n_aug_ + 1);

    //create vector for predicted state
    VectorXd x = VectorXd(n_x_);

    //create covariance matrix for prediction
    MatrixXd P = MatrixXd(n_x_, n_x_);


    /*
    // set weights
    double weight_0 = lambda_ / (lambda_ + n_aug_);

    weights(0) = weight_0;
    for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
        double weight = 0.5 / (n_aug_ + lambda_);
        weights(i) = weight;
    }
*/
    //predicted state mean
    x.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        x = x + weights(i) * Xsig_pred.col(i);
    }

    //predicted state covariance matrix
    P.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

        // state difference
        VectorXd x_diff = Xsig_pred.col(i) - x;
        //angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        P = P + weights(i) * x_diff * x_diff.transpose();
    }

    //write result
    x_ = x;
    P_ = P;
    //std::cout << "x = " << std::endl << x << std::endl;
    //std::cout << "P = " << std::endl << P << std::endl;
}

void UKF::PredictRadarMeasurement(VectorXd *z_out, MatrixXd *S_out, MatrixXd &Zsig, MatrixXd &Xsig_pred) {
    //cout << "PredictRadarMeasurement" << endl;

    //cout << "Xsig_pred" << Xsig_pred << endl;
    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
        //cout << "loop: " << Xsig_pred.rows() << " : " << Xsig_pred.cols() << endl;
        // extract values for better readibility
        double p_x = Xsig_pred(0, i);
        double p_y = Xsig_pred(1, i);
        double v = Xsig_pred(2, i);
        double yaw = Xsig_pred(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // measurement model
        //cout << "measurement model" << endl;
        Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                        //r
        Zsig(1, i) = atan2(p_y, p_x);                                 //phi
        Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);   //r_dot
    }
    //cout << "z_pred" << endl;
    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z_radar);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z_radar, n_z_radar);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z_radar, n_z_radar);
    R << std_radr_ * std_radr_, 0, 0,
            0, std_radphi_ * std_radphi_, 0,
            0, 0, std_radrd_ * std_radrd_;
    S = S + R;

    //write result
    *z_out = z_pred;
    *S_out = S;
}

void UKF::Init(const MeasurementPackage &measurement_pack) {
    // Intialize with first measurement
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        /**
        Convert radar from polar to cartesian coordinates and initialize state.
         ro, phi, ro_dot
        */
        double rho = measurement_pack.raw_measurements_[0]; // range
        double phi = measurement_pack.raw_measurements_[1]; // bearing

        // Coordinates conversion from polar to cartesian
        double px = rho * cos(phi);
        double py = rho * sin(phi);

        /**
         * Note that although radar does include velocity information, the radar velocity
         * and the CTRV velocity are not the same. Radar velocity is measured from
         * the autonomous vehicle's perspective. If you drew a straight line from
         * the vehicle to the bicycle, radar measures the velocity along that line.
         * In the CTRV model, the velocity is from the object's perspective,
         * which in this case is the bicycle; the CTRV velocity is tangential
         * to the circle along which the bicycle travels. Therefore, you cannot
         * directly use the radar velocity measurement to initialize the state vector.
         */
        double rho_dot = measurement_pack.raw_measurements_[2]; // velocity of rho
        double vx = rho_dot * cos(phi);
        double vy = rho_dot * sin(phi);
        x_ << px, py, rho_dot, vx, vy;

    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
        //set the state with the initial location and zero velocity
        x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0, 0;
    } else {
        throw "Unknown sensor type";
    }

    previous_timestamp_ = measurement_pack.timestamp_;
    initial_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
}
