using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Text;
using System.IO;

//¹Ø½Úµã£¨¹Ç÷À£©±àºÅ
//  0-Hip
//  1-RightHip
//  2-RightKnee
//  3-RightAnkle
//  4-LeftHip
//  5-LeftKnee
//  6-LeftAnkle
//  7-Spine
//  8-Chest
//  9-Thorax
//  10-Neck
//  11-Head
//  12-LeftShoulder
//  13-LeftUpArm
//  14-LeftElbow
//  15-LeftWrist
//  16-RightShoulder
//  17-RightUpArm
//  18-RightElbow
//  19-RightWrist

//  20-RightAnkleEndSite
//  21-LeftAnkleEndSite
//  22-HeadEndSite
//  23-LeftWristEndSite
//  24-RightWristEnsSite

public class LocalCoordinate
{
    public Vector3 x;
    public Vector3 y;
    public Vector3 z;
    public LocalCoordinate(Vector3 _x, Vector3 _y, Vector3 _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }
}

public class BvhHandler
{
    private int numJoints;  //total number of joints
    private int rootJoint;  //define the root joint, default number 0, i.e.the hip
    private bool[] isEndJoint;   //define the end joint
    private string[] jointName_blender; //define joint name for blender skeleton
    private List<List<int>> children;  //define chidren joints for each joint
    private int[] parent; //define parent joint for each joint
    private Vector3[] TPoseDirection; //define direction for each bone under T-pose
    private int numFrames;  //total nFrames of video
    private float frameRate=30;  //defined by user, default 30fps
    private int[] blender_order; //map blender skeleton to videopose3d skeleton
    private Vector3 globalX;  
    private Vector3 globalY; 
    private Vector3 globalZ;  //global coordinate system,used for calculating localRotation
    string bvhText;

    public BvhHandler()
    {
        numJoints = 20;
        rootJoint = 0;
        isEndJoint = new bool[numJoints+5];
        for (int i = 0; i < numJoints+5; i++)
        {
            if (i>=20)
                isEndJoint[i] = true;
            else
                isEndJoint[i] = false;
        }
        jointName_blender = new string[20] { "Hip", "RightHip", "RightKnee", "RightAnkle", "LeftHip", "LeftKnee", "LeftAnkle", "Spine", "Chest","Thorax", "Neck", "Head", "LeftShoulder","LeftUpArm", "LeftElbow", "LeftWrist", "RightShoulder", "RightUpArm","RightElbow", "RightWrist" };
        children = new List<List<int>> { new List<int>{1,4,7},
                                         new List<int>{2},
                                         new List<int>{3},
                                         new List<int>{20},
                                         new List<int>{5},
                                         new List<int>{6},
                                         new List<int>{21},
                                         new List<int>{8},
                                         new List<int>{9},
                                         new List<int>{10,12,16},
                                         new List<int>{11},
                                         new List<int>{22},
                                         new List<int>{13},
                                         new List<int>{14},
                                         new List<int>{15},
                                         new List<int>{23},
                                         new List<int>{17},
                                         new List<int>{18},
                                         new List<int>{19},
                                         new List<int>{24},
                                         new List<int>{ },
                                         new List<int>{ },
                                         new List<int>{ },
                                         new List<int>{ },
                                         new List<int>{ }};
        parent = new int[20] { -1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 10, 9, 12, 13, 14, 9, 16, 17,18};
        blender_order = new int[17] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        frameRate = 30;
        globalX = new Vector3(1, 0, 0);
        globalY = new Vector3(0, 1, 0);
        globalZ = new Vector3(0, 0, 1);
    }

    //calculate the length of each bone,used for the 'offset' field in BVH header
    private float[] calculateBoneLength(Vector3[] poses)
    {
        float[] lengs = new float[numJoints];
        //´ý²¹³ä

        return lengs;
    }

    //calculate the initial offset of each bone, default in T-pose
    private Vector3[] calculateInitialOffset(float[] lengs)
    {
        Vector3[] offset = new Vector3[20] {new Vector3(0,0,0),
                                            new Vector3(1,0,0),
                                            new Vector3(0,0,-2.0f),
                                            new Vector3(0,0,-1.8f),
                                            new Vector3(-1,0,0),
                                            new Vector3(0,0,-2.0f),
                                            new Vector3(0,0,-1.8f),
                                            new Vector3(0,0,1),
                                            new Vector3(0,0,1),
                                            new Vector3(0,0,1),
                                            new Vector3(0,0,1),
                                            new Vector3(0,0,1),
                                            new Vector3(-1,0,0),
                                            new Vector3(-0.3f,0,0),
                                            new Vector3(-1.2f,0,0),
                                            new Vector3(-1.2f,0,0),
                                            new Vector3(1,0,0),
                                            new Vector3(0.3f,0,0),
                                            new Vector3(1.2f,0,0),
                                            new Vector3(1.2f,0,0)};
        return offset;
    }

    //convert vector coordinate system from VideoPose3D  to Unity
    private Vector3 P2U(Vector3 v)
    {
        return v.x * (new Vector3(-1, 0, 0)) + v.y * (new Vector3(0, -1, 0)) + v.z * (new Vector3(0, 0, -1));
    }

    //convert vector coordinate system from Unity  to Blender
    //private Vector3 U2B(Vector3 v)
    //{
    //    return v.x * (new Vector3(0, -1, 0)) + v.y * (new Vector3(0, 0, 1)) + v.z * (new Vector3(1, 0, 0));
    //}
    private Vector3 U2B(Vector3 v)
    {
        return v.x * (new Vector3(1, 0, 0)) + v.y * (new Vector3(0, 0, 1)) + v.z * (new Vector3(0, 1, 0));
    }

    //calculate local rotation of each joint in each frame according to the 3D coordination
    //local rotation is indicated by Euler angles in the order of Z-X-Y
    private float[,] pose2Euler(Vector3[] poses,string skeleton_type)
    {
        int[] order = new int[numJoints];
        if (skeleton_type == "blender")
        {
            for (int i = 0; i < blender_order.Length; i++)
                order[i] = blender_order[i];
        }

        float[,] localTransInfo = new float[numFrames, 3 * numJoints + 3];
        Vector3[] localRotationPerFrame = new Vector3[numJoints];
        LocalCoordinate[] localCoordinates = new LocalCoordinate[numJoints];

        for (int frame = 0; frame < numFrames; frame++)
        {
            int nJoints = 17;
            //Firstly,we need to determine the local coordination system for each joint
            //Here is an implementation for right-hand coordination system
            //For Hip(index=0)
            Vector3 x= poses[frame * nJoints + 1] - poses[frame * nJoints + 4];
            Vector3 a= poses[frame * nJoints + 4] - poses[frame * nJoints + 7];
            Vector3 b = poses[frame * nJoints + 1] - poses[frame * nJoints + 7];
            Vector3 forward = Vector3.Cross(a, b);
            Vector3 up = Vector3.Cross(forward, x);
            Vector3 left = Vector3.Cross(up, forward);
            localCoordinates[0] = new LocalCoordinate(U2B(left), U2B(forward),  U2B(up));

            //For RightHip(index=1)
            int parentIndex = 0;
            int nowIndex = 1;
            int childIndex = 2;
            a = poses[frame * nJoints + nowIndex] - poses[frame * nJoints + childIndex];
            b = poses[frame * nJoints + parentIndex] - poses[frame * nJoints + nowIndex];
            forward = Vector3.Cross(a, b);
            up = poses[frame * nJoints + nowIndex] - poses[frame * nJoints + childIndex];
            left = Vector3.Cross(up, forward);
            localCoordinates[1] = new LocalCoordinate(U2B(left), U2B(forward), U2B(up));

            //For RightKnee(index=2)
            parentIndex = 1;
            nowIndex = 2;
            childIndex = 3;
            a = poses[frame * nJoints + nowIndex] - poses[frame * nJoints + childIndex];
            b = poses[frame * nJoints + parentIndex] - poses[frame * nJoints + nowIndex];
            x = Vector3.Cross(b, a);
            up = poses[frame * nJoints + nowIndex] - poses[frame * nJoints + childIndex];
            forward = Vector3.Cross(x, up);
            left = Vector3.Cross(up, forward);
            localCoordinates[2] = new LocalCoordinate(U2B(left), U2B(forward), U2B(up));

            //For RightAnkle(index=3)
            //note that foot's position is not involved in the VideoPose3D estimation model
            //yet we can estimate the rotation of ankle using the position of hip, knee and ankle
            parentIndex = 2;
            a = poses[frame * nJoints + 1] - poses[frame * nJoints + 2];
            b = poses[frame * nJoints + 3] - poses[frame * nJoints + 2];
            Vector3 temp0 = Vector3.Cross(a, b);
            Vector3 temp1= poses[frame * nJoints + 1] - poses[frame * nJoints + 3];
            forward = Vector3.Cross(temp1 , temp0);
            forward.y = 0;
            up = Vector3.up;
            left = Vector3.Cross(up, forward);
            localCoordinates[3] = new LocalCoordinate(U2B(left), U2B(forward), U2B(up));

            //For LeftHip(index=4)
            parentIndex = 0;
            nowIndex = 4;
            childIndex = 5;
            a = poses[frame * nJoints + nowIndex] - poses[frame * nJoints + childIndex];
            b = poses[frame * nJoints + parentIndex] - poses[frame * nJoints + nowIndex];
            forward = Vector3.Cross(b, a);
            up = poses[frame * nJoints + nowIndex] - poses[frame * nJoints + childIndex];
            left = Vector3.Cross(up, forward);
            localCoordinates[4] = new LocalCoordinate(U2B(left), U2B(forward), U2B(up));

            //For LeftKnee(index=5)
            parentIndex = 4;
            nowIndex = 5;
            childIndex = 6;
            a = poses[frame * nJoints + nowIndex] - poses[frame * nJoints + childIndex];
            b = poses[frame * nJoints + parentIndex] - poses[frame * nJoints + nowIndex];
            x = Vector3.Cross(b, a);
            up = poses[frame * nJoints + nowIndex] - poses[frame * nJoints + childIndex];
            forward = Vector3.Cross(x, up);
            left = Vector3.Cross(up, forward);
            localCoordinates[5] = new LocalCoordinate(U2B(left), U2B(forward), U2B(up));

            //For LeftAnkle(index=6)
            parentIndex = 5;
            nowIndex = 6;
            a = poses[frame * nJoints + 4] - poses[frame * nJoints + 5];
            b = poses[frame * nJoints + 6] - poses[frame * nJoints + 5];
            temp0 = Vector3.Cross(a, b);
            temp1 = poses[frame * nJoints + 4] - poses[frame * nJoints + 6];
            forward = Vector3.Cross(temp1, temp0);
            forward.y = 0;
            up = Vector3.up;
            left = Vector3.Cross(up, forward);
            localCoordinates[6] = new LocalCoordinate(U2B(left), U2B(forward), U2B(up));

            //For Spine(index=7)
            parentIndex = 0;
            nowIndex = 7;
            a = poses[frame * nJoints + 14] - poses[frame * nJoints + 7];
            b = poses[frame * nJoints + 11] - poses[frame * nJoints + 7];
            forward = Vector3.Cross(a, b);
            up= poses[frame * nJoints + 8] - poses[frame * nJoints + 7];
            left = Vector3.Cross(up, forward);
            localCoordinates[7] = new LocalCoordinate(U2B(left), U2B(forward), U2B(up));

            //For Chest(index=8)
            //actually,chest is not involved in VideoPose3D skeleton, 
            //add it, just to provide more freedom for editors to operate on generated poses
            nowIndex = 8;
            localCoordinates[8] = localCoordinates[7];

            //For Thorax(index=9)
            //calculation for thorax is similar to that for Hip, both have multiple children joints
            parentIndex = 8;
            nowIndex = 9;
            x = poses[frame * nJoints + 14] - poses[frame * nJoints + 11];
            a = poses[frame * nJoints + 11] - poses[frame * nJoints + 9];
            b = poses[frame * nJoints + 14] - poses[frame * nJoints + 9];
            forward = Vector3.Cross(a, b);
            up = Vector3.Cross(forward, x);
            left = Vector3.Cross(up, forward);
            localCoordinates[9] = new LocalCoordinate(U2B(left), U2B(forward), U2B(up));

            //For Neck(index=10)
            //simply set it to identity
            nowIndex = 10;
            localCoordinates[10] = localCoordinates[9];

            //For HeadEndSite(index=11)
            //Since eye position is not involved in VideoPose3D model, simply make head'forward vector consistent with that of chest 
            parentIndex = 10;
            nowIndex = 11;
            localCoordinates[11] = localCoordinates[10];
            //a = poses[frame * numJoints + 11] - poses[frame * numJoints + 9];
            //b = poses[frame * numJoints + 14] - poses[frame * numJoints + 9];
            //forward = Vector3.Cross(a, b);
            //up= poses[frame * numJoints + parentIndex] - poses[frame * numJoints + nowIndex];
            //left = Vector3.Cross(up, forward);
            //localCoordinates[10] = new LocalCoordinate(U2B(left), U2B(forward), U2B(up));

            //For LeftShoulder(index=12)
            localCoordinates[12] = localCoordinates[9];

            //For LeftUpArm(index=13)
            a = poses[frame * nJoints + 8] - poses[frame * nJoints + 11];
            b = poses[frame * nJoints + 11] - poses[frame * nJoints + 12];
            up = Vector3.Cross(a, b);
            x = poses[frame * nJoints + 11] - poses[frame * nJoints + 12];
            forward = Vector3.Cross(x, up);
            left = Vector3.Cross(up, forward);
            localCoordinates[13] = new LocalCoordinate(U2B(left), U2B(forward), U2B(up));

            //For LeftElbow(index=14)
            a = poses[frame * nJoints + 11] - poses[frame * nJoints + 12];
            b = poses[frame * nJoints + 12] - poses[frame * nJoints + 13];
            up = Vector3.Cross(a, b);
            x = poses[frame * nJoints + 12] - poses[frame * nJoints + 13];
            forward = Vector3.Cross(x, up);
            left = Vector3.Cross(up, forward);
            localCoordinates[14] = new LocalCoordinate(U2B(left), U2B(forward), U2B(up));

            //For LeftWrist(index=15)
            //Since hand position is not involved in VideoPose3D model, simply set its local rotation to identity
            localCoordinates[15] = localCoordinates[14];

            //For RightShoulder(index=16)
            localCoordinates[16] = localCoordinates[9];

            //For RightUpArm(index=17)
            a = poses[frame * nJoints + 8] - poses[frame * nJoints + 14];
            b = poses[frame * nJoints + 14] - poses[frame * nJoints + 15];
            up = Vector3.Cross(b, a);
            x = poses[frame * nJoints + 15] - poses[frame * nJoints + 14];
            forward = Vector3.Cross(x, up);
            left = Vector3.Cross(up, forward);
            localCoordinates[17] = new LocalCoordinate(U2B(left), U2B(forward), U2B(up));


            //For RightElbow(index=18)
            a = poses[frame * nJoints + 14] - poses[frame * nJoints + 15];
            b = poses[frame * nJoints + 15] - poses[frame * nJoints + 16];
            up = Vector3.Cross(b, a);
            x = poses[frame * nJoints + 16] - poses[frame * nJoints + 15];
            forward = Vector3.Cross(x, up);
            left = Vector3.Cross(up, forward);
            localCoordinates[18] = new LocalCoordinate(U2B(left), U2B(forward), U2B(up));


            //For RightWrist(index=19)
            localCoordinates[19] = localCoordinates[18];


            //Normalize all vectors
            for (int joint = 0; joint < numJoints; joint++)
            {
                localCoordinates[joint].x=Vector3.Normalize(localCoordinates[joint].x);
                localCoordinates[joint].y=Vector3.Normalize(localCoordinates[joint].y);
                localCoordinates[joint].z=Vector3.Normalize(localCoordinates[joint].z);
            }

            //Then, we need to calculate the localRotation in the form of Euler Angle for each joint
            for (int joint = 0; joint < numJoints; joint++)
            {
                //local rotation of each joint is determined by both its own coordinate and its parent's coordinate
                LocalCoordinate coord_child = localCoordinates[joint];
                LocalCoordinate coord_parent;
                if (joint == 0)       //For root joint,set its parent to the global coordinate system
                    coord_parent = new LocalCoordinate(globalX, globalY, globalZ);
                else
                    coord_parent = localCoordinates[parent[joint]];
                float[,] dcm = getDCM(coord_parent,coord_child);  //calculate DCM according to local coordinate system
                localRotationPerFrame[joint] = getEuler(dcm);     //calculate Euler Angles in order Z-X-Y
            }


            //store channel information for a frame
            //Firstly, write POSITION CHANNEL of the root joint
            localTransInfo[frame, 0] = 0;
            localTransInfo[frame, 1] = 0;
            localTransInfo[frame, 2] = 0;
            for (int joint = 0; joint < numJoints; joint++)
            {
               localTransInfo[frame, 3 + 3 * joint] = localRotationPerFrame[joint].z;
               localTransInfo[frame, 3 + 3 * joint + 1] = localRotationPerFrame[joint].x;
               localTransInfo[frame, 3 + 3 * joint + 2] = localRotationPerFrame[joint].y;
            }
        }
        return localTransInfo;
    }

    //calculate DCM(Direction Cosine Matrix)
    private float[,] getDCM(LocalCoordinate parent, LocalCoordinate child)
    {
        float[,] dcm = new float[3,3];
        dcm[0, 0] = Vector3.Dot(child.x, parent.x);
        dcm[1, 0] = Vector3.Dot(child.x, parent.y);
        dcm[2, 0] = Vector3.Dot(child.x, parent.z);

        dcm[0, 1] = Vector3.Dot(child.y, parent.x);
        dcm[1, 1] = Vector3.Dot(child.y, parent.y);
        dcm[2, 1] = Vector3.Dot(child.y, parent.z);

        dcm[0, 2] = Vector3.Dot(child.z, parent.x);
        dcm[1, 2] = Vector3.Dot(child.z, parent.y);
        dcm[2, 2] = Vector3.Dot(child.z, parent.z);

        return dcm;
    }

    private Vector3 getEuler(float[,] dcm)
    {
        float angleZ = Mathf.Atan2(-dcm[0, 1],dcm[1, 1])*Mathf.Rad2Deg;
        float angleX = Mathf.Asin(dcm[2, 1]) * Mathf.Rad2Deg;
        float angleY = Mathf.Atan2(-dcm[2, 0],dcm[2, 2]) * Mathf.Rad2Deg;

        return new Vector3(angleX, angleY, angleZ);
    }
    //write BVH header recursively
    private string writeBvhHeader(string[] jointName,Vector3[] initialOffset,int currentJoint, int level)
    {
        string text = "";
        string space = new string(' ', 4*level);
        text += space;
        if (currentJoint == rootJoint)
            text += "HIERARCHY\n"+"ROOT " + jointName[currentJoint] + "\n";
        else if (isEndJoint[currentJoint])
            text += "End Site \n";
        else
            text += "Joint " + jointName[currentJoint] + "\n";
        text += space + "{\n";
        space += new string(' ', 4);
        if (isEndJoint[currentJoint])
            text+=space+"OFFSET 0 0 0";
        else
        {
            text += space + "OFFSET " + initialOffset[currentJoint].x.ToString() + " " + initialOffset[currentJoint].y.ToString() + " " + initialOffset[currentJoint].z.ToString() + "\n";
            text += space + "CHANNELS ";
        }
        if (currentJoint == rootJoint)
            text += "6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n";
        else if (isEndJoint[currentJoint])
            text += "\n";
        else 
            text+= "3 Zrotation Xrotation Yrotation\n";

        for (int i = 0; i < children[currentJoint].Count; i++)
            text += writeBvhHeader(jointName,initialOffset, children[currentJoint][i], level + 1);
        space= new string(' ', 4 * level);
        text += space + "}\n";
        return text;

    }

    //write BVH channels
    private string writeChannels(string text,float[,]transInfo)
    {
        text += "MOTION\n";
        text += "Frames: " + numFrames.ToString()+ "\n";
        text += "Frame Time: " + (1.0f / frameRate).ToString() + "\n";
        for (int frame = 0; frame < transInfo.GetLength(0); frame++)
        {
            for (int i = 0; i < transInfo.GetLength(1); i++)
                text += transInfo[frame, i].ToString()+" ";
            text += "\n";
        }
        return text;
    }

    //temp
    private Vector3[] reOrder(Vector3[] _poses)
    {
        numFrames = _poses.Length / (numJoints-3);
        Vector3[] poses = new Vector3[_poses.Length];
        Debug.Log(numFrames);
        Debug.Log(poses.Length);
        for (int frame = 0; frame < numFrames; frame++)
        {
            for(int joint=0;joint<numJoints-3;joint++)
                poses[frame*(numJoints-3)+joint]= _poses[frame * (numJoints-3) + blender_order[joint]];
        }
        return poses;
    }

    private Vector3[] convertToUnity(Vector3[] poses)
    {
        for (int i = 0; i < poses.Length; i++)
            poses[i] = P2U(poses[i]);
        return poses;
    }

    private Vector3[] FilterAndInterpolate(Vector3[] poses, float filter,int numInter)
    {
        for (int joint = 0; joint < 17; joint++)
        {
            for (int frame = 1; frame < numFrames; frame++)
            {
                poses[frame * 17 + joint] = filter * poses[(frame - 1) * 17 + joint] + (1 - filter) * poses[frame * 17 + joint];
            }
        }
        Vector3[] nPoses = new Vector3[((numFrames - 1) * (1 + numInter) + 1)*17];
        for (int frame = 0; frame < numFrames-1; frame++)
        {
            for (int joint = 0; joint < 17; joint++)
            {
                Vector3 D = poses[(frame + 1) * 17 + joint] - poses[frame * 17 + joint];
                Vector3 d = D / (numInter + 1);
                nPoses[((1 + numInter) * frame + 0) * 17 + joint] = poses[frame * 17 + joint];
                for(int i=1;i<=numInter;i++)
                    nPoses[((1 + numInter) * frame + i) * 17 + joint] = poses[frame * 17 + joint]+d*i;
            }
        }
        numFrames = (numFrames - 1) * (1 + numInter) + 1;
        return nPoses;
    }

    //save poses as BVH file
    public void createBVH(Vector3[] _poses, float rate,float filter,int numInter)
    {
        Vector3[] poses=reOrder(_poses);
        poses = FilterAndInterpolate(poses,filter,numInter);
        poses = convertToUnity(poses);
        frameRate = rate;
        float[] lengs = calculateBoneLength(poses);
        Vector3[] initialOffset = calculateInitialOffset(lengs);
        float[,] transInfo = pose2Euler(poses,"blender");
        string text =writeBvhHeader(jointName_blender,initialOffset, 0, 0);
        bvhText = writeChannels(text,transInfo);
    }

    public void saveBVH(string savePath)
    {
        FileStream fs = new FileStream(savePath, FileMode.Create);
        StreamWriter sw = new StreamWriter(fs);
        sw.Write(bvhText);
        sw.Flush();
        sw.Close();
    }
}
