using AICircleDetector.AI;
using Force.Crc32;
using OneOf.Types;
using ProtoBuf;
using SkiaSharp;
using System;
using System.Drawing;
using System.Threading;
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

                int label = GetLabelFromAnnotation(annotationPath, classMap);

                using Bitmap bmp = new Bitmap(imagePath);
                using Bitmap resized = new Bitmap(bmp, new Size(AIConfig.ImageSize, AIConfig.ImageSize));
                byte[] imageBytes = ImageToByteArray(resized);

                Feature imageFeature = TfFeatureHelper.Bytes(imageBytes);
                Feature labelFeature = TfFeatureHelper.Int64(label);

                Features features = new Features();
                features.feature.Add("image", imageFeature);
                features.feature.Add("label", labelFeature);

                Example example = new Example
                {
                    Features = features
                };

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


        public static int GetLabelFromAnnotation(string annotationPath, Dictionary<int, string> classMap)
        {
            //Console.WriteLine($"Processing annotation: {annotationPath}");

            try
            {
                // Load the XML
                var doc = XDocument.Load(annotationPath);

                // Count all <object> elements — assuming each <object> represents a circle
                int count = doc.Descendants("object").Count();

                //Console.WriteLine($"Extracted label (number of circles): {count}");
                return count;
            }
            catch (Exception ex)
            {
                //Console.WriteLine($"Error reading annotation: {ex.Message}");
                return 0; // Fallback
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
