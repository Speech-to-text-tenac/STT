import classesp from "./Prediction.module.css";
import axios from "axios";
import React, { useState } from "react";
// import { trackPromise } from 'react-promise-tracker';// let gumStream = null;
import { usePromiseTracker } from "react-promise-tracker";
import {ThreeDots} from 'react-loader-spinner';
// let audioContext = null;
// const audio;
const LoadingIndicator = props => {
  const { promiseInProgress } = usePromiseTracker();

  return promiseInProgress && 
    <div
      style={{
        width: "100%",
        height: "100",
        display: "flex",
        justifyContent: "center",
        alignItems: "center"
      }}
    >
      <ThreeDots type="ThreeDots" color="#2BAD60" height="100" width="100" />
    </div>
};
export const Prediction = () => {
  // const [show, setShow] = useState(false);
  // const [view, setView] = useState(true);
  const [userData, setUserData] = useState({});

  // const startRecording = () => {
  //   let constraints = {
  //     audio: true,
  //     video: false,
  //   };

  //   audioContext = new window.AudioContext();
  //   console.log("sample rate: " + audioContext.sampleRate);

  //   navigator.mediaDevices
  //     .getUserMedia(constraints)
  //     .then(function (stream) {
  //       console.log("initializing Recorder.js ...");

  //       gumStream = stream;

  //       let input = audioContext.createMediaStreamSource(stream);
  //       console.log(input);

  //       recorder = new window.Recorder(input, {
  //         numChannels: 1,
  //       });

  //       console.log(recorder);
  //       recorder.record();
  //       console.log("Recording started");
  //     })
  //     .catch(function (err) {
  //       //enable the record button if getUserMedia() fails
  //       console.log(err);
  //     });
  // };

  // const stopRecording = () => {
  //   console.log("stopButton clicked");

  //   recorder.stop(); //stop microphone access
  //   gumStream.getAudioTracks()[0].stop();

  //   recorder.exportWAV(onStop);
  // };

  // const onStop = (blob) => {
  //   console.log("uploading...");

  //   let data = new FormData();

  //   data.append("text", "this is the transcription of the audio file");
  //   data.append("wavfile", blob, "recording.wav");

  //   // const config = {
  //   //   headers: { "content-type": "multipart/form-data" },
  //   // };
  //   console.log(data);
  //   // axios.post("http://localhost:8080/asr/", data, config);
  // };

  const [file, setFile] = useState();

  function handleChange(event) {
    setFile(event.target.files[0]);
  }
  
  function handleSubmit(event) {
    // trackPromise(
    event.preventDefault()
    const url = "https://stt-amharic.azurewebsites.net/predict";
    const formData = new FormData();
    formData.append("file", file);
    formData.append("fileName", file.name);
    console.log(file.name);
    const config = {
      headers: {
        "content-type": "multipart/form-data",
      },
    };
    
    axios.post(url, formData, config).then((response) => {
      console.log(response.data.message);
      setUserData(response.data);
     

    }).catch((err)=>{
      console.log(err,"error here")
    });
  }
  

  return (
    <>
      <div className={classesp.container}>
         <div className={classesp.upload}>
              <div>
                <h2>Upload Audio</h2>
                <div>
                  <form onSubmit={handleSubmit}>                    
                    <input type="file" onChange={handleChange} />
                    <button type="submit">Upload</button>
                  </form>
                </div>
              </div>
              <section>
              <LoadingIndicator/>
              <p>{userData.message}</p>
              </section>
            </div>


        {/* <div>
          {view ? (
            <div className={classesp.display}>
              <div>
                <main>
                  <h2>Record Audio</h2>
                  <button onClick={startRecording} type="button">
                    Start
                  </button>
                  <button onClick={stopRecording} type="button">
                    Stop
                  </button>
                </main>
                <br></br>
                <br></br>
                <audio autoPlay controls></audio>
              </div>
              <section>
                <p>{userData.message}</p>

              </section>
            </div>
          ) : (
            <div className={classesp.upload}>
              <div>
                <h2>Upload Audio</h2>
                <div>
                  <form onSubmit={handleSubmit}>                    
                    <input type="file" onChange={handleChange} />
                    <button type="submit">Upload</button>
                  </form>
                </div>
              </div>
              <section>
                <p>ትንበያዉ </p>
                <p>{userData.message}</p>
              </section>
            </div>
          )}
        </div> */}
      </div>
    </>
  );
};
