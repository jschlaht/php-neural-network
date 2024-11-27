<?php

namespace App\Tests\Service;

use App\Service\NeuralNetwork;
use PHPUnit\Framework\TestCase;

class NeuralNetworkTest extends TestCase
{
    public function testNeuralNetwork() {
        $neuralNetwork = new NeuralNetwork(3,3,3,0.3);
        $this->assertInstanceOf(NeuralNetwork::class, $neuralNetwork);
        $this->assertSame($neuralNetwork->getInputNodes(), 3);
        $this->assertSame($neuralNetwork->getHiddenNodes(), 3);
        $this->assertSame($neuralNetwork->getOutputNodes(), 3);
        $this->assertSame($neuralNetwork->getLearningRate(), 0.3);
    }

    public function testWeightsMatrixValues() {
        $numberOfInputNodes = 3;
        $numberOfHiddenNodes = 5;
        $numberOfOutputNodes = 2;
        $learningRate = 0.3;
        $neuralNetwork = new NeuralNetwork($numberOfInputNodes, $numberOfHiddenNodes, $numberOfOutputNodes, $learningRate);
        $weightsInputToHidden = $neuralNetwork->getWeightsInputToHidden();
        $weightsHiddenToOutout = $neuralNetwork->getWeightsHiddenToOutput();

        $this->assertEquals(count($weightsInputToHidden), $numberOfHiddenNodes);
        $this->assertEquals(count($weightsHiddenToOutout), $numberOfOutputNodes);

        $this->assertEquals(count($weightsInputToHidden[0]), $numberOfInputNodes);
        $this->assertEquals(count($weightsHiddenToOutout[0]), $numberOfHiddenNodes);

        $min = min(array_map('min', $weightsInputToHidden));
        $max = max(array_map('max', $weightsInputToHidden));

        $this->assertTrue(min(array_map('min', $weightsInputToHidden)) > -1);
        $this->assertTrue(max(array_map('max', $weightsInputToHidden)) < 1);
        $this->assertTrue(min(array_map('min', $weightsHiddenToOutout)) > -1);
        $this->assertTrue(max(array_map('max', $weightsHiddenToOutout)) < 1);
    }

}
