using AICircleDetector.AI;
using Force.Crc32;
using Google.Protobuf;
using OneOf.Types;
using ProtoBuf;
using SkiaSharp;
using System;
using System.Drawing;
using System.Threading;
using System.Xml;
using System.Xml.Linq;

namespace AICircleDetector.AI
{
    public static class TrainingDataBuilder
    {
        private static readonly Random _rand = new();

        public static async Task<TrainingDataBuilderResult> CreateTrainingData(int imageCount = 20)
        {
            try
            {
                string _currentGUID = Guid.NewGuid().ToString();

                Console.WriteLine($"CreateTrainingData {_currentGUID} has started, please wait ...");

                string _currentSessionDir = Path.Combine(Environment.CurrentDirectory, AIConfig.TrainingFolderName, _currentGUID);
                string _currentImageDir = Path.Combine(_currentSessionDir, AIConfig.ImageFolderName);
                string _currentAnnotationDir = Path.Combine(_currentSessionDir, AIConfig.AnnotationsFolderName);

                string labelMapPath = Path.Combine(_currentSessionDir, AIConfig.LabelMapName);

                string trainListPath = Path.Combine(_currentSessionDir, AIConfig.TrainListName);
                string valListPath = Path.Combine(_currentSessionDir, AIConfig.ValListName);
                string trainvalListPath = Path.Combine(_currentSessionDir, AIConfig.TrainValListName);


                Directory.CreateDirectory(_currentImageDir!);
                Directory.CreateDirectory(_currentAnnotationDir!);

                for (int i = 0; i < imageCount; i++)
                {
                    int ringCount = _rand.Next(AIConfig.MinCircles, AIConfig.MaxCircles + 1);
                    string fileName = $"log_{i:D3}.png";
                    string imagePath = Path.Combine(_currentImageDir!, fileName);

                    var circles = GenerateRingImage(imagePath, ringCount);
                    SaveAnnotationXml(_currentAnnotationDir, fileName, circles);
                }

                CreateTrainValFiles(imageCount, trainListPath, valListPath, trainvalListPath, ".png", 0.2);

                CreateLabelMap(labelMapPath!);
                var classMap = ParseLabelMap(labelMapPath);

                CreateSerializedTFRecord(trainListPath, _currentImageDir, _currentAnnotationDir, classMap, AIConfig.TrainDataName);
                CreateSerializedTFRecord(valListPath, _currentImageDir, _currentAnnotationDir, classMap, AIConfig.ValDataName);

                Console.WriteLine($"CreateTrainingData {_currentGUID} finished!");

                return new TrainingDataBuilderResult
                {
                    Success = true,
                    Message = $"Successfully generated {imageCount} training images.",
                    OutputDirectory = _currentSessionDir!
                };
            }
            catch (Exception ex)
            {
                return new TrainingDataBuilderResult
                {
                    Success = false,
                    Message = $"Error during generation: {ex.Message}",
                    OutputDirectory = string.Empty
                };
            }
        }


        public static Dictionary<int, string> ParseLabelMap(string labelMapPath)
        {
            // Example logic to parse label map (adjust as needed)
            var classMap = new Dictionary<int, string>();

            foreach (var line in File.ReadLines(labelMapPath))
            {
                if (line.Contains("id:"))
                {
                    int id = int.Parse(line.Split(':')[1].Trim());
                    string name = File.ReadLines(labelMapPath).SkipWhile(x => !x.Contains("name:")).First().Split(':')[1].Trim().Trim('"');
                    classMap.Add(id, name);
                }
            }

            return classMap;
        }

        public static Example CreateExample(byte[] imageBytes, int width, int height, List<BoundingBox> boxes)
        {
            var xmin = boxes.Select(b => (float)b.XMin / width).ToList();
            var ymin = boxes.Select(b => (float)b.YMin / height).ToList();
            var xmax = boxes.Select(b => (float)b.XMax / width).ToList();
            var ymax = boxes.Select(b => (float)b.YMax / height).ToList();

            var classesText = boxes.Select(b => "circle").ToList();  // list of strings
            var classes = boxes.Select(b => 1L).ToList();            // long values for labels

            var features = new Features();

            // Image bytes feature
            features.feature.Add("image/encoded", TfFeatureHelper.Bytes(imageBytes));

            // Image size
            features.feature.Add("image/height", TfFeatureHelper.Int64(height));
            features.feature.Add("image/width", TfFeatureHelper.Int64(width));

            // Bounding box coordinates as floats
            features.feature.Add("image/object/bbox/xmin", TfFeatureHelper.FloatList(xmin));
            features.feature.Add("image/object/bbox/ymin", TfFeatureHelper.FloatList(ymin));
            features.feature.Add("image/object/bbox/xmax", TfFeatureHelper.FloatList(xmax));
            features.feature.Add("image/object/bbox/ymax", TfFeatureHelper.FloatList(ymax));

            // Class text as bytes list
            features.feature.Add("image/object/class/text", TfFeatureHelper.BytesList(classesText));

            // Class labels as int64 list
            features.feature.Add("image/object/class/label", TfFeatureHelper.Int64List(classes));

            return new Example { Features = features };
        }



        public static void CreateSerializedTFRecord(string dataListPath, string imagesPath, string annotationsPath, Dictionary<int, string> classMap, string tfRecordFileName)
        {
            var imageFilenames = File.ReadAllLines(dataListPath);
            string tfRecordFilePath = Path.Combine(Path.GetDirectoryName(dataListPath), tfRecordFileName);

            using var writer = new FileStream(tfRecordFilePath, FileMode.Create, FileAccess.Write);

            foreach (var filename in imageFilenames)
            {
                string imagePath = Path.Combine(imagesPath, filename);
                string annotationPath = Path.Combine(annotationsPath, Path.ChangeExtension(filename, ".xml"));

                if (!File.Exists(imagePath) || !File.Exists(annotationPath))
                {
                    Console.WriteLine($"Skipping missing file(s): {filename}");
                    continue;
                }

                // Parse bounding boxes from annotation XML (you'll need a helper method for this)
                List<BoundingBox> boxes = ParseBoundingBoxesFromXml(annotationPath, classMap);

                using Bitmap bmp = new Bitmap(imagePath);
                using Bitmap resized = new Bitmap(bmp, new Size(AIConfig.ImageSize, AIConfig.ImageSize));
                byte[] imageBytes = ImageToByteArray(resized);

                // Create the Example with bounding boxes, normalized coords and image bytes
                Example example = CreateExample(imageBytes, resized.Width, resized.Height, boxes);

                // Serialize Example to byte array
                byte[] exampleBytes;
                using (var ms = new MemoryStream())
                {
                    Serializer.Serialize(ms, example);
                    exampleBytes = ms.ToArray();
                }

                // Write TFRecord formatted record
                WriteTFRecord(writer, exampleBytes);
            }

            Console.WriteLine($"{tfRecordFileName} created at {tfRecordFilePath}");
        }

        private static List<BoundingBox> ParseBoundingBoxesFromXml(string xmlFilePath, Dictionary<int, string> classMap = null)
        {
            var boxes = new List<BoundingBox>();
            var doc = new XmlDocument();
            doc.Load(xmlFilePath);

            var objectNodes = doc.SelectNodes("//object");
            if (objectNodes == null) return boxes;

            foreach (XmlNode objNode in objectNodes)
            {
                var nameNode = objNode.SelectSingleNode("name");
                if (nameNode == null || nameNode.InnerText != "circle")
                    continue;  // Skip non-circle objects

                var bndboxNode = objNode.SelectSingleNode("bndbox");
                if (bndboxNode == null)
                    continue;

                int xmin = int.Parse(bndboxNode.SelectSingleNode("xmin").InnerText);
                int ymin = int.Parse(bndboxNode.SelectSingleNode("ymin").InnerText);
                int xmax = int.Parse(bndboxNode.SelectSingleNode("xmax").InnerText);
                int ymax = int.Parse(bndboxNode.SelectSingleNode("ymax").InnerText);

                boxes.Add(new BoundingBox(xmin, ymin, xmax, ymax));
            }

            return boxes;
        }

        private static void WriteTFRecord(Stream stream, byte[] data)
        {
            // Length (ulong)
            ulong length = (ulong)data.Length;
            byte[] lengthBytes = BitConverter.GetBytes(length);
            stream.Write(lengthBytes, 0, 8);

            // Length CRC
            uint lengthCrc = MaskedCrc32c(lengthBytes);
            byte[] lengthCrcBytes = BitConverter.GetBytes(lengthCrc);
            stream.Write(lengthCrcBytes, 0, 4);

            // Data
            stream.Write(data, 0, data.Length);

            // Data CRC
            uint dataCrc = MaskedCrc32c(data);
            byte[] dataCrcBytes = BitConverter.GetBytes(dataCrc);
            stream.Write(dataCrcBytes, 0, 4);
        }

        // Implementation of masked CRC32c for TFRecord
        private static uint MaskedCrc32c(byte[] data)
        {
            uint crc = Crc32Algorithm.Compute(data);
            return ((crc >> 15) | (crc << 17)) + 0xa282ead8;
        }


        public static (int count, List<int> diameters) GetLabelFromAnnotation(string annotationPath, Dictionary<int, string> classMap)
        {
            try
            {
                var doc = XDocument.Load(annotationPath);

                var objects = doc.Descendants("object").ToList();
                int count = objects.Count;

                List<int> diameters = new List<int>();

                foreach (var obj in objects)
                {
                    var bndbox = obj.Element("bndbox");
                    if (bndbox != null)
                    {
                        int xmin = int.Parse(bndbox.Element("xmin")?.Value ?? "0");
                        int ymin = int.Parse(bndbox.Element("ymin")?.Value ?? "0");
                        int xmax = int.Parse(bndbox.Element("xmax")?.Value ?? "0");
                        int ymax = int.Parse(bndbox.Element("ymax")?.Value ?? "0");

                        int width = xmax - xmin;
                        int height = ymax - ymin;
                        int diameter = (width + height) / 2;
                        diameters.Add(diameter);
                    }
                }

                return (count, diameters);
            }
            catch
            {
                return (0, new List<int>());
            }
        }



        public static byte[] ImageToByteArray(Bitmap image)
        {
            using (var ms = new MemoryStream())
            {
                image.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
                return ms.ToArray();
            }
        }


        private static List<(float x, float y, float r)> GenerateRingImage(string filePath, int ringCount)
        {
            const int maxAttempts = 1000;
            var placedCircles = new List<(float x, float y, float r)>();

            using SKBitmap bitmap = new SKBitmap(AIConfig.ImageSize, AIConfig.ImageSize);
            using SKCanvas canvas = new SKCanvas(bitmap);
            canvas.Clear(SKColors.White);

            int attempts = 0;
            while (placedCircles.Count < ringCount && attempts < maxAttempts)
            {
                float radius = _rand.Next(5, AIConfig.ImageSize / 4);
                float x = _rand.Next((int)radius, AIConfig.ImageSize - (int)radius);
                float y = _rand.Next((int)radius, AIConfig.ImageSize - (int)radius);

                bool collides = placedCircles.Any(c =>
                {
                    float dx = c.x - x;
                    float dy = c.y - y;
                    float distSq = dx * dx + dy * dy;
                    float minDist = c.r + radius + 1;
                    return distSq < minDist * minDist;
                });

                if (collides)
                {
                    attempts++;
                    continue;
                }

                placedCircles.Add((x, y, radius));

                using SKPaint paint = new SKPaint
                {
                    Style = SKPaintStyle.Stroke,
                    Color = RandomGreenBrown(),
                    StrokeWidth = _rand.Next(1, 5),
                    IsAntialias = true
                };

                canvas.DrawCircle(x, y, radius, paint);
            }

            using SKImage image = SKImage.FromBitmap(bitmap);
            using SKData data = image.Encode(SKEncodedImageFormat.Png, 100);
            using FileStream stream = File.OpenWrite(filePath);
            data.SaveTo(stream);

            return placedCircles;
        }

        private static void SaveAnnotationXml(string _currentAnnotationDir, string imageFileName, List<(float x, float y, float r)> circles)
        {
            string imagePath = Path.Combine("images", imageFileName);
            string annotationPath = Path.Combine(_currentAnnotationDir!, Path.ChangeExtension(imageFileName, ".xml"));

            var annotation = new XElement("annotation",
                new XElement("folder", "images"),
                new XElement("filename", imageFileName),
                new XElement("path", imagePath),
                new XElement("source",
                    new XElement("database", "Unknown")),
                new XElement("size",
                    new XElement("width", AIConfig.ImageSize),
                    new XElement("height", AIConfig.ImageSize),
                    new XElement("depth", 3)),
                new XElement("segmented", 0)
            );

            foreach (var (x, y, r) in circles)
            {
                int xmin = (int)(x - r);
                int ymin = (int)(y - r);
                int xmax = (int)(x + r);
                int ymax = (int)(y + r);

                annotation.Add(
                    new XElement("object",
                        new XElement("name", "circle"),
                        new XElement("pose", "Unspecified"),
                        new XElement("truncated", 0),
                        new XElement("difficult", 0),
                        new XElement("bndbox",
                            new XElement("xmin", xmin),
                            new XElement("ymin", ymin),
                            new XElement("xmax", xmax),
                            new XElement("ymax", ymax)
                        )
                    )
                );
            }

            XDocument xmlDoc = new(annotation);
            xmlDoc.Save(annotationPath);
        }

        private static SKColor RandomGreenBrown()
        {
            int r = _rand.Next(60, 160);
            int g = _rand.Next(80, 180);
            int b = _rand.Next(30, 100);
            return new SKColor((byte)r, (byte)g, (byte)b);
        }

        private static void CreateLabelMap(string labelMapPath)
        {
            var content = "item {\n  id: 1\n  name: 'circle'\n}\n";
            File.WriteAllText(labelMapPath, content);
        }

        private static void CreateTrainValFiles(int imageCount, string trainListPath, string valListPath, string trainvalListPath, string fileExtension, double validationSplit)
        {
            var allFilenames = Enumerable.Range(0, imageCount)
                                         .Select(i => $"log_{i:D3}{fileExtension}")
                                         .ToList();

            // Shuffle the filenames to ensure randomness
            var shuffledFilenames = allFilenames.OrderBy(x => _rand.Next()).ToList();

            // Split the shuffled list into train and val based on the split ratio (80% train, 20% val)
            int valCount = (int)(imageCount * validationSplit);  // 20% for validation
            var valFilenames = shuffledFilenames.Take(valCount).ToList();
            var trainFilenames = shuffledFilenames.Skip(valCount).ToList();

            // Write to trainval.txt (all filenames)
            File.WriteAllLines(trainvalListPath, allFilenames);

            // Write to train.txt (80% for training)
            File.WriteAllLines(trainListPath, trainFilenames);

            // Write to val.txt (20% for validation)
            File.WriteAllLines(valListPath, valFilenames);
        }


    }
}
