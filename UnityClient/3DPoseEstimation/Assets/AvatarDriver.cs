using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEngine.UI;

//Providing interactions in the Visualization interface for users 
public class AvatarDriver : MonoBehaviour
{
    private string bvhPath;
    public GameObject avatar;
    public Animator anim;
    public AnimationHandler animHandler;
    private GameObject textTip;
    public GameObject textTipMain;
    private GameObject panelVisual;
    private GameObject panelMain;
    private GameObject buttonVisual;
    public InputField inputSpeed;
    private GameObject camController;
    private int tipCnt = -1;
    private float[] rates;
    private int ratePointer = 3;
    private int cnt = 0;

    // Start is called before the first frame update
    void Start()
    {
        rates = new float[7] { 0.5f, 0.75f, 0.8f, 1, 1.2f, 1.5f, 2 };
        camController = GameObject.Find("CameraController");
        avatar = GameObject.FindGameObjectWithTag("boy");
        anim = avatar.GetComponent<Animator>();
        animHandler = new AnimationHandler();
        animHandler.setAvatar(anim, "Armature", GameObject.Find("Hips").transform);
        //animHandler.setAvatar(anim);
        //inputSpeed.DeactivateInputField();
        //inputSpeed.interactable = false;
        textTip = GameObject.Find("textTip");
        panelVisual = GameObject.Find("panelVisual");
        panelMain = GameObject.Find("panelMain");
        buttonVisual = GameObject.Find("visualControl");
        inputSpeed.GetComponent<InputField>().text = "1.0";
        Invoke("delaySetActive", 0.01f);    //Delay the action to SetActive to ensure that other scripts have assigned the variable.
    }


    // Update is called once per frame
    void Update()
    {

    }

    public void visualControl()
    {
        panelVisual.SetActive(!panelVisual.active);
    }

    public void moveCam()
    {
        camController.GetComponent<CameraController>().setPosition(camController.GetComponent<CameraController>().getPosition() - new Vector3(0, 0, 1.8f));
        cnt += 1;
        if (cnt == 180)
        {
            CancelInvoke("moveCam");
            //textTipMain.SetActive(true);
            panelMain.SetActive(true);
            camController.GetComponent<CameraController>().setEnable(false);
            cnt = 0;
        }
    }

    public void toMain()
    {
        panelVisual.SetActive(false);
        textTip.SetActive(false);
        GameObject.Find("Curtain").GetComponent<MeshRenderer>().material.SetColor("_Color", new Color(151 / 255, 179 / 255, 253 / 255, 100/255));
        InvokeRepeating("moveCam", 0, 0.006f);

    }
    public void delaySetActive()
    {
        panelVisual.SetActive(false);
    }

    public void hideTip()
    {
        if (tipCnt == 0)
            textTip.SetActive(false);
        tipCnt -= 1;
    }

    //Operations for controlling the animation
    public void playAnimation()
    {
        if (!animHandler.Ready())
        {
            tipCnt += 1;
            textTip.GetComponent<Text>().text = "Please select a BVH file first!";
            textTip.SetActive(true);
            Invoke("hideTip", 3.0f);
            return;
        }
        animHandler.Play();
    }

    public void resetAnimation()
    {
        animHandler.Reset();
    }

    public void pauseAnimation()
    {
        if (!animHandler.Ready())
        {
            tipCnt += 1;
            textTip.GetComponent<Text>().text = "Please select a BVH file first!";
            textTip.SetActive(true);
            Invoke("hideTip", 3.0f);
            return;
        }
        animHandler.Pause();
    }

    public void continueAnimation()
    {
        if (!animHandler.Ready())
        {
            tipCnt += 1;
            textTip.GetComponent<Text>().text = "Please select a BVH file first!";
            textTip.SetActive(true);
            Invoke("hideTip", 3.0f);
            return;
        }
        animHandler.Continue();
    }

    public void speedUp()
    {
        if (ratePointer == rates.Length - 1)
            return;
        ratePointer += 1;
        animHandler.setSpeed(rates[ratePointer]);
        inputSpeed.GetComponent<InputField>().text =rates[ratePointer].ToString();
    }

    public void speedDown()
    {
        if (ratePointer == 0)
            return;
        ratePointer -= 1;
        animHandler.setSpeed(rates[ratePointer]);
        inputSpeed.GetComponent<InputField>().text = rates[ratePointer].ToString();
    }

    public void freeControl(float value)
    {
        if (!animHandler.Ready())
        {
            tipCnt += 1;
            textTip.GetComponent<Text>().text = "Please select a BVH file first!";
            textTip.SetActive(true);
            Invoke("hideTip", 3.0f);
            return;
        }
        if (animHandler.isPlaying())
        {
            tipCnt += 1;
            textTip.GetComponent<Text>().text = "Please pause or reset the animation first!";
            textTip.SetActive(true);
            Invoke("hideTip", 3.0f);
            return;
        }
        animHandler.setToTime(value);
    }
    public void selectFile()
    {
        bvhPath = EditorUtility.OpenFilePanel("Select a BVH file to parse", "D:/", "bvh");
        if (bvhPath.Length == 0)
        {
            tipCnt += 1;
            textTip.GetComponent<Text>().text = "File path could not be empty !";
            textTip.SetActive(true);
            Invoke("hideTip", 3.0f);
            return;
        }
        if (!animHandler.parseFile(bvhPath))
        {
            textTip.GetComponent<Text>().text = "BVH format error !";
            return;
        }
        animHandler.Load();
    }
}
