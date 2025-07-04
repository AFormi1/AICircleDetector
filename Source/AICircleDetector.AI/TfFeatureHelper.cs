using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;


namespace AICircleDetector.AI
{

    public static class TfFeatureHelper
    {
        // For Bytes, we're dealing with byte arrays
        public static Feature Bytes(byte[] value) =>
            new Feature
            {
                BytesList = new BytesList
                {
                    Values = { value }  // Directly add byte[]
                }
            };




        // For Int64, we convert a single long value to an array with one item
        public static Feature Int64(long value) =>
            new Feature { Int64List = new Int64List { Values = new long[] { value } } };


        // For Int64List, we convert IEnumerable<long> to long[]
        public static Feature Int64List(IEnumerable<long> values) =>
                    new Feature { Int64List = new Int64List { Values = values.ToArray() } };

        // For FloatList, convert IEnumerable<float> to float[]
        public static Feature FloatList(IEnumerable<float> values) =>
            new Feature { FloatList = new FloatList { Values = values.ToArray() } };


        public static Feature BytesList(IEnumerable<string> values)
        {
            var feature = new Feature();
            var bytesList = new BytesList();
            foreach (var v in values)
            {
                bytesList.Values.Add(Encoding.UTF8.GetBytes(v));  // directly add byte[]
            }
            feature.BytesList = bytesList;
            return feature;
        }

    }
}
