CustomerMD5Key - Key associated to the customer

b SCID - Broker ID

+c SelectedPackage - Policy type

+c FirstDriverMaritalStatus - Marital Status of the First registered driver

+f CarAnnualMileage - Car annual mile age

+c CarFuelId - Kind of fuel used

+c CarUsageId - The reason to use the vehicle business or pleasure or both or anything else

+f FirstDriverAge - Age of the first registered driver

+f CarInsuredValue - Value of the car insured

+f CarAge - Age of the insured car

effacer FirstDriverDrivingLicenseNumberY -  First Driver Driving License number

+f VoluntaryExcess -  AXA do not cover all the cost for an accident for example the customer may select e.g. that (s)he does not mind paying 100 euros and 
AXA will cover anything above that or the customer may select 50, 1000, 2000 etc

+c CarParkingTypeId - Garage or off street etc there are approximately 5-10 different options

+f PolicyHolderNoClaimDiscountYears -  How many years the customer did not have any accidents 

+c FirstDriverDrivingLicenceType -  Full car, Provisional Car, EU provisional, Oversees full licence, Provisional  Car but full motorbike licence etc. more 
than 20 different categories

+c CoverIsNoClaimDiscountSelected - the driver had an accident the last one year

+c CarDrivingEntitlement -  full, provisional etc.

+c CarTransmissionId -  manual, automatic etc. 

en attente / important
SocioDemographicId -  areas that the customer resides and education level, i.e. Single Phd education living in a town working in technology

en attente / important
? PolicyHolderResidencyArea - ? 

+f AllDriversNbConvictions - Number of convictions for all drivers

en attente / ???????
RatedDriverNumber - Which driver is the highest risk

+c IsPolicyholderAHomeowner -   Home owner

en attente / important
CarMakeId - internal car code that provides us information about the car i.e. Toyota Auris 1.2 cc automatic, Fiat Passat 1.5cc manual etc.

+f DaysSinceCarPurchase - clear enough

+c NameOfPolicyProduct - clear enough

en attente / important
AffinityCodeId  - similar to Broker ID


UK

Q: pk change FirstDriverDrivingLicenseNumberY

Q: Volontary Excess

Q: CoverIsNoClaimDiscountSelected

Quel pays ? Driver licence type





cols = ['SCID',
        'SelectedPackage',
        'FirstDriverMaritalStatus',
        'CarAnnualMileage',
        'CarFuelId',
        'CarUsageId',
        'FirstDriverAge',
        'CarInsuredValue',
        'CarAge',
        'FirstDriverDrivingLicenseNumberY',
        'VoluntaryExcess',
        'CarParkingTypeId',
        'PolicyHolderNoClaimDiscountYears',
        'FirstDriverDrivingLicenceType',
        'CoverIsNoClaimDiscountSelected',
        'CarDrivingEntitlement',
        'CarTransmissionId',
        'SocioDemographicId',
        'PolicyHolderResidencyArea',
        'AllDriversNbConvictions',
        'RatedDriverNumber',
        'IsPolicyholderAHomeowner',
        'DaysSinceCarPurchase',
        'NameOfPolicyProduct',
        'AffinityCodeId']


cols = ['SCID',
        'SelectedPackage', 
        'FirstDriverMaritalStatus',
        'CarAnnualMileage',
        'CarFuelId',
        'CarUsageId',
        'FirstDriverAge',
        'CarInsuredValue',
        'CarAge',
        'FirstDriverDrivingLicenseNumberY',
        'VoluntaryExcess',
        'CarParkingTypeId',
        'PolicyHolderNoClaimDiscountYears',
        'FirstDriverDrivingLicenceType',
        'CoverIsNoClaimDiscountSelected',
        'CarDrivingEntitlement',
        'CarTransmissionId',
        'SocioDemographicId',
        'PolicyHolderResidencyArea',
        'AllDriversNbConvictions',
        'RatedDriverNumber',
        'IsPolicyholderAHomeowner',
        'DaysSinceCarPurchase',
        'NameOfPolicyProduct',
        'AffinityCodeId']