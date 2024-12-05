<?php

namespace App\Tests\Service;

use App\Service\NeuralNetwork;
use PHPUnit\Framework\TestCase;

class NeuralNetworkTest extends TestCase
{
    public function testNeuralNetwork()
    {
        $neuralNetwork = new NeuralNetwork(3,3,3,0.3);
        $this->assertInstanceOf(NeuralNetwork::class, $neuralNetwork);
        $this->assertSame($neuralNetwork->getInputNodes(), 3);
        $this->assertSame($neuralNetwork->getHiddenNodes(), 3);
        $this->assertSame($neuralNetwork->getOutputNodes(), 3);
        $this->assertSame($neuralNetwork->getLearningRate(), 0.3);
    }

    public function testWeightsMatrixValues()
    {
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

        #$min = min(array_map('min', $weightsInputToHidden));
        #$max = max(array_map('max', $weightsInputToHidden));

        $this->assertTrue(min(array_map('min', $weightsInputToHidden)) > -1);
        $this->assertTrue(max(array_map('max', $weightsInputToHidden)) < 1);
        $this->assertTrue(min(array_map('min', $weightsHiddenToOutout)) > -1);
        $this->assertTrue(max(array_map('max', $weightsHiddenToOutout)) < 1);
    }

    public function testDotProductUnitMatrix()
    {
        $neuralNetwork = new NeuralNetwork(3,3,3,0.3);

        $matrix1 = array(
            array(1, 0),
            array(0, 1),
        );

        $matrix2 = array(
            array(4, 1),
            array(2, 2),
        );

        $result = $neuralNetwork->dotProduct($matrix1, $matrix2);

        $this->assertIsArray($result);
        $this->assertSame($matrix2, $result);
    }

    public function testDotProductSimpleExample()
    {
        $neuralNetwork = new NeuralNetwork(3,3,3,0.3);

        $matrix1 = array(
            array(1, 2, 3),
            array(4, 5, 6),
        );

        $matrix2 = array(
            array(7, 8),
            array(9, 10),
            array(11, 12),
        );

        $matrix3 = array(
            array(58, 64),
            array(139, 154),
        );

        $result = $neuralNetwork->dotProduct($matrix1, $matrix2);

        $this->assertIsArray($result);
        $this->assertSame($matrix3, $result);
    }
    public function testDotProductWrongDimensions()
    {
        $neuralNetwork = new NeuralNetwork(3,3,3,0.3);

        $matrix1 = [
            [1, 2, 3],
            [4, 5, 6],
        ];

        $matrix2 = [
            [7, 8, 9],
            [10, 11, 12],
        ];

        $this->expectExceptionMessage("Matrix dimensions do not match");
        $this->expectException(\Exception::class);
        $neuralNetwork->dotProduct($matrix1, $matrix2);
    }

    public function testActivateList()
    {
        $neuralNetwork = new NeuralNetwork(3,3,3,0.3);

        $matrix1 = array(1.16, 0.42, 0.62);
        $matrix2 = array(
            array(0.761),
            array(0.603),
            array(0.650),
        );

        $result = $neuralNetwork->doActivate($matrix1);
        $this->assertIsArray($result);
        $this->assertSame($matrix2, $result);
    }
    public function testActivateArray()
    {
        $neuralNetwork = new NeuralNetwork(3,3,3,0.3);

        $matrix1 = array(
            array(1.16),
            array(0.42),
            array(0.62),
        );
        $matrix2 = array(
            array(0.761),
            array(0.603),
            array(0.650),
        );

        $result = $neuralNetwork->doActivate($matrix1);
        $this->assertIsArray($result);
        $this->assertSame($matrix2, $result);
    }

    public function testQueryMethod()
    {
        $neuralNetwork = new NeuralNetwork(3,3,3,0.3);
        $inputList = array(1.0, 0.5, -1.5);
        $result = $neuralNetwork->query($inputList);

        $this->assertIsArray($result);
    }
    public function testTrainMethod()
    {
        $neuralNetwork = new NeuralNetwork(3,3,3,0.3);
        $inputList = array(1.0, 0.5, -1.5);
        $targetList = array(2.0, 1.5, -0.5);
        $neuralNetwork->train($inputList, $targetList);
    }

    public function testCalculatingDifference()
    {
        $neuralNetwork = new NeuralNetwork(3,3,3,0.3);
        $matrix1 = array(
            array(10),
            array(15),
            array(20),
        );
        $matrix2 = array(
            array(8),
            array(11),
            array(15),
        );
        $matrix3 = array(
            array(2),
            array(4),
            array(5),
        );

        $result = $neuralNetwork->doCalculateDifference($matrix1, $matrix2);
        $this->assertSame($matrix3, $result);
    }
}
