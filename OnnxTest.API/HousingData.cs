using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxTest.API
{
    public class HousingData
    {
        public float MedianIncome { get; }
        public float MedianHouseAge { get; }
        public float AverageNumberOfRooms { get; }
        public float AverageNumberOfBedrooms { get; }
        public float Population { get; }
        public float AverageOccupancy { get; }
        public float Latitude { get; }
        public float Longitude { get; }

        public HousingData(float mi, float mha, float anr, float anb, float ao, float pop, float lat, float lng)
        {
            MedianIncome = mi;
            MedianHouseAge = mha;
            AverageNumberOfRooms = anr;
            AverageNumberOfBedrooms = anb;
            Population = pop;
            AverageOccupancy = ao;
            Latitude = lat;
            Longitude = lng;


        }

        public Tensor<float> AsTensor()
        {
            float[] data = new float[]
            {
            MedianIncome, MedianHouseAge, AverageNumberOfRooms, AverageNumberOfBedrooms,
            Population, AverageOccupancy, Latitude, Longitude
            };
            int[] dimensions = new int[] { 1, 8 };
            return new DenseTensor<float>(data, dimensions);
        }
    }
}
