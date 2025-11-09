using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AICircleDetector.AI
{
    public class CircleAnnotation
    {
        public string Label { get; set; } = "circle";
        public float XMin { get; set; }
        public float YMin { get; set; }
        public float XMax { get; set; }
        public float YMax { get; set; }
    }
}
