import classesp from "./Prediction.module.css";
import axios from "axios";
import React, { useState } from "react";
// let gumStream = null;
// let recorder = null;
// let audioContext = null;
// const audio;

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
    event.preventDefault();
    const url = "https://stt-amharic.herokuapp.com/predict";
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
      console.log("here");
      console.log(response);
      console.log(response.data.message);
      setUserData(response.data);
      // let audio = response.data.message;
      // this.setState({audioMessage:audio})
      // console.log(audioMessage)

    }).catch((err)=>{
      console.log(err,"error here")
    });
  }

  return (
    <>
      <div className={classesp.container}>
        {/* <div className={classesp.link}>
          <span title="choose" onClick={() => setShow(!show)}></span>
          <br></br>
        <span></span>
          {show ? (
            <ul>
              <li>
                <Link to="" onClick={() => setView(true)}>
                  Record Audio
                </Link>
              </li>
              <hr></hr>
              <li>
                <Link to="" onClick={() => setView(false)}>
                  Upload Audio
                </Link>
              </li>
            </ul>
          ) : null}
        </div> */}
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
