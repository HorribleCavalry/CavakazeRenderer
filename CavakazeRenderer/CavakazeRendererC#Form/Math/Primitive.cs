using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CavakazeRenderer.Math
{
    abstract class Primitive
    {
    }

    class Triangle
    {
        public Vec3[] points;
        public Vec2[] uvs
        public Triangle()
        {
            points = new Vec3[3];
            uvs = new Vec2[3];
        }
    }
}
