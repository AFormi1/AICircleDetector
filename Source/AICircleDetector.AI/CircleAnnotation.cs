using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AICircleDetector.AI
{
    public class CircleAnnotation
    {
        public string filename { get; set; }
        public List<BoundingBox> circles { get; set; }
    }
}
