using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.Windows.Forms;

using CavakazeRenderer.CrossPlatformAPIManager;
using CavakazeRenderer.Managers;

namespace CavakazeRenderer
{
    public partial class CavakazeRendererMainForm : Form
    {
        private RenderManager renderManager;
        private Bitmap renderImage;
        public CavakazeRendererMainForm()
        {
            InitializeComponent();
            CrossPlatformAPIManager.CudaAPI.OpenDebugConsole();
            renderManager = new RenderManager();
            renderImage = new Bitmap(RenderImage.Width, RenderImage.Height);
            RenderImage.Image = renderImage;
        }

        private void flowLayoutPanel1_Paint(object sender, PaintEventArgs e)
        {

        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void label2_Click(object sender, EventArgs e)
        {

        }

        private void treeView1_AfterSelect(object sender, TreeViewEventArgs e)
        {

        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void label4_Click(object sender, EventArgs e)
        {

        }

        private void label6_Click(object sender, EventArgs e)
        {

        }

        private void AddButton_Click(object sender, EventArgs e)
        {
            Hierarchy.SelectedNode.Nodes.Add("A");
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Hierarchy.SelectedNode.Remove();
        }

        private void RenderButton_Click(object sender, EventArgs e)
        {
            //renderManager.StartRendering(renderImage.Width,renderImage.Height,renderImage.GetHbitmap());
            renderManager.StartRendering(256,144,renderImage.GetHbitmap());
            RenderImage.Update();
        }
    }
}
