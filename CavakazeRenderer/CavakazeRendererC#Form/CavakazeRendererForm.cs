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
        private System.Drawing.Imaging.BitmapData renderImage_data;
        private Bitmap renderImage;

        unsafe public CavakazeRendererMainForm()
        {
            InitializeComponent();
            CrossPlatformAPIManager.CudaAPI.OpenDebugConsole();
            renderImage = new Bitmap(RenderImage.Width, RenderImage.Height);
            renderImage_data = renderImage.LockBits(new System.Drawing.Rectangle(0, 0, renderImage.Width, renderImage.Height), System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            renderImage.UnlockBits(renderImage_data);
            RenderImage.Image = renderImage;

            ImageParam imageParam = new ImageParam();
            imageParam.width = RenderImage.Width;
            imageParam.height = RenderImage.Height;
            imageParam.imagePtr = renderImage_data.Scan0.ToPointer();
            renderManager = new RenderManager(imageParam, ref RenderImage);
        }

        private void UpdateImageCallBack(object state)
        {
            RenderImage.Refresh();
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
            renderManager.StartRendering();
            UpdateTicker.Start();
        }

        private void UpdateTicker_Tick(object sender, EventArgs e)
        {
            RenderImage.Refresh();
        }
    }
}
