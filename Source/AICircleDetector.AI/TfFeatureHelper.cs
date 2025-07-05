using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using Tensorflow;
using static Google.Protobuf.Compiler.CodeGeneratorResponse.Types;


namespace AICircleDetector.AI
{

    public static class TfFeatureHelper
    {
        // For Bytes, we're dealing with byte arrays
        public static Feature Bytes(byte[] value) =>
            new Feature { BytesList = new BytesList { Values = { Google.Protobuf.ByteString.CopyFrom(value).ToByteArray() } } };

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
                // Convert ByteString to byte[] using ToByteArray()
                bytesList.Values.Add(Google.Protobuf.ByteString.CopyFromUtf8(v).ToByteArray());
            }
            feature.BytesList = bytesList;
            return feature;
        }

        public static (List<float> xmins, List<float> xmaxs, List<float> ymins, List<float> ymaxs, List<string> labelsText, List<long> labelsIdx)
            ParseAnnotation(string annotationPath, Dictionary<string, int> labelMap)
        {
            var doc = XDocument.Load(annotationPath);
            var objects = doc.Descendants("object");

            var xmins = new List<float>();
            var xmaxs = new List<float>();
            var ymins = new List<float>();
            var ymaxs = new List<float>();
            var labelsText = new List<string>();
            var labelsIdx = new List<long>();

            foreach (var obj in objects)
            {
                var name = obj.Element("name")?.Value.Trim();
                var labelIdx = labelMap[name!];

                var bndbox = obj.Element("bndbox");
                float xmin = float.Parse(bndbox.Element("xmin")?.Value!) / AIConfig.ImageSize.Width;
                float ymin = float.Parse(bndbox.Element("ymin")?.Value!) / AIConfig.ImageSize.Height;
                float xmax = float.Parse(bndbox.Element("xmax")?.Value!) / AIConfig.ImageSize.Width;
                float ymax = float.Parse(bndbox.Element("ymax")?.Value!) / AIConfig.ImageSize.Height;

                xmins.Add(xmin);
                xmaxs.Add(xmax);
                ymins.Add(ymin);
                ymaxs.Add(ymax);
                labelsText.Add(name);
                labelsIdx.Add(labelIdx);
            }

            return (xmins, xmaxs, ymins, ymaxs, labelsText, labelsIdx);
        }


    }
}
