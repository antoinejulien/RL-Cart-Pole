#include <MPU6050_light.h>
#include "SerialCommunication.h"
#include "Wire.h"

/*--------------------------- Global variables ------------------------------*/
MPU6050 mpu(Wire);
SerialCommunication serialComm;

/*-------------------------- Function headers -------------------------------*/
void getMPU6050angles(float angles[3]);

/*------------------------------- Functions ---------------------------------*/
/*
 * Get the angles from the MPU6050
 * @Param: angles array of float containing the angles
 */
void getMPU6050angles(float angles[3])
{
  mpu.update();
  angles[0] = mpu.getAngleX();
  angles[1] = mpu.getAngleY();
  angles[2] = mpu.getAngleZ();
}

/*-------------------------------- Setup ------------------------------------*/
void setup() {
  Serial.setTimeout(5);
  Serial.begin(115200);

  Wire.begin();
  byte status = mpu.begin();

  // Retry connecting until connection is established with MPU if it did not work first time
  while (status != 0) {
    status = mpu.begin();
    delay(2000);
  }

  // Uncomment this line if the MPU6050 is mounted upside-down
  //mpu.upsideDownMounting = true; 

  // MPU6050 gyroscope and accelerometer
  mpu.calcOffsets();
  delay(1000);
} 

/*--------------------------------- Loop ------------------------------------*/
void loop() {
  float motorCommand;
  float mpuAngles[3];

  if(Serial.available() > 0) 
  {
//    if(serial.serialDecoder(commands) == 0)
//    {
//      moveMotors(commands[0], commands[1], commands[2]);
//    }
//    else
//    {
//      //Update IMU angles before sending the values
//      OCRimu_.updateAngles();
//      OCRimu_.getAngles(OCRimu_angles);
//
//      //mpu.update();
//      //getMPU6050angles(MPU6050angles);
//      
//      serial.serialEncoder(OCRimu_angles);   
//    }
  }

  getMPU6050angles(mpuAngles);

}
