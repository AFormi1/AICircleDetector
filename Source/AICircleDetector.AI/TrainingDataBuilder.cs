using Force.Crc32;
using ProtoBuf;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Xml.Linq;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace AICircleDetector.AI
{
    public class TrainingDataBuilder
    {
        public static string TrainingModelFullURL => Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TrainedModel");
        private static string ImageFolder => Path.Combine(TrainingModelFullURL, "Images");
        private static string AnnotationFolder => Path.Combine(TrainingModelFullURL, "Annotations");

        
    }
}
