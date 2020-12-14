using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CavakazeRenderer.Common
{
    class Vec2
    {
        public float x, y;
    }

    class Vec3
    {
        public float x, y, z;
    }

    class Vec4
    {
        public float x, y, z, w;
    }

    class Quaternion
    {
        public float x, y, z, w;
    }

    class Transform
    {
        public Vec3 Position;
        public Quaternion Rotation;
        public Vec3 Scale;
    }
}
