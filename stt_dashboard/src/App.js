// import './App.css';
import Navbar from 'react-bootstrap/Navbar';
import Nav from "react-bootstrap/Nav";

function App() {
  return (
    <div className="App">
      <header className="App-header">
          
        <Navbar bg="dark" variant="dark">
          
          <Navbar.Brand href="#home"></Navbar.Brand>
          <Nav className="me-auto">
            <Nav.Link href="#home">Home</Nav.Link>
            <Nav.Link href="#features">Visuals</Nav.Link>
            <Nav.Link href="#pricing">Upload your audio</Nav.Link>
          </Nav>
          
        </Navbar>
  

      </header>
    </div>
  );
}

export default App;
