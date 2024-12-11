<?php

namespace App\Tests\Service;

use App\Service\NeuralNetwork;
use PHPUnit\Framework\TestCase;

class NeuralNetworkWeights352Test extends TestCase
{
    private array $weightsInputToHiddden = [
        [0.9, 0.3, 0.4],
        [0.2, 0.8, 0.2],
        [0.1, 0.5, 0.6],
        [0.6, 0.3, 0.8],
        [0.7, 0.1, 0.3],
    ];
    private array $weightsInputToHidddenChangeOne = [
        [0.9, 0.3, 0.4],
        [0.2, 0.8, 0.2],
        [0.1, 0.5, 0.6],
        [0.6, 0.3, 0.8],
        [0.7, 0.1, 0.3],
    ];
    private array $deltaWeightsInputToHiddden = [
        [0.9, 0.3, 0.4],
        [0.2, 0.8, 0.2],
        [0.1, 0.5, 0.6],
        [0.6, 0.3, 0.8],
        [0.7, 0.1, 0.3],
    ];
    private array $weightsHiddenToOutput = [
        [0.3, 0.7, 0.8, 0.5, 0.4],
        [0.6, 0.5, 0.2, 0.1, 0.9],
    ];

    private array $weightsHiddenToOutputChangeOne = [
        [0.302, 0.701, 0.801, 0.502, 0.401],
        [0.607, 0.506, 0.206, 0.107, 0.907],
    ];
    private array $deltaWeightsHiddenToOutput = [
        [-0.0018, -0.0012, -0.0015, -0.0018, -0.0015],
        [-0.0075, -0.006, -0.0063, -0.0075, -0.0069],
    ];

    private array $expectedHiddenInputs = [
        [1.16],
        [0.42],
        [0.62],
        [1.21],
        [0.88],
    ];
    private array $expectedHiddenOutputs = [
        [0.761],
        [0.603],
        [0.650],
        [0.770],
        [0.707],
    ];
    private array $expectedFinalInputs = [
        [1.838],
        [1.601],
    ];
    private array $expectedFinalOutputs = [
        [0.863],
        [0.832],
    ];
    private array $expectedOutputErrors = [
        [-0.063],
        [-0.232],
    ];
    private array $expectedHiddenErrors = [
        [-0.158],
        [-0.160],
        [-0.097],
        [-0.055],
        [-0.234],
    ];
    private int $inputNodes = 3;
    private int $hiddenNodes = 5;
    private int $outputNodes = 2;
    private float $learningRate = 0.3;

    public function testTrain()
    {
        $neuralNetwork = new NeuralNetwork($this->inputNodes, $this->hiddenNodes, $this->outputNodes, $this->learningRate);

        $inputList = array(0.9, 0.1, 0.8);
        $targetList = array(0.8, 0.6);

        $neuralNetwork->setWeightsInputToHidden($this->weightsInputToHiddden);
        $neuralNetwork->setWeightsHiddenToOutput($this->weightsHiddenToOutput);

        $neuralNetwork->train($inputList, $targetList);

        $this->assertSame($this->expectedHiddenInputs, $neuralNetwork->getHiddenInputs());
        $this->assertSame($this->expectedHiddenOutputs, $neuralNetwork->getHiddenOutputs());
        $this->assertSame($this->expectedFinalInputs, $neuralNetwork->getFinalInputs());
        $this->assertSame($this->expectedFinalOutputs, $neuralNetwork->getFinalOutputs());
        $this->assertSame($this->expectedOutputErrors, $neuralNetwork->getOutputErrors());
        $this->assertSame($this->expectedHiddenErrors, $neuralNetwork->getHiddenErrors());
        $this->assertSame($this->deltaWeightsHiddenToOutput, $neuralNetwork->getDeltaWeightsHiddenToOutput());
        $this->assertSame($this->deltaWeightsInputToHiddden, $neuralNetwork->getDeltaWeightsInputToHidden());
        $this->assertSame($this->weightsHiddenToOutputChangeOne, $neuralNetwork->getWeightsHiddenToOutput());
        $this->assertSame($this->weightsInputToHidddenChangeOne, $neuralNetwork->getWeightsInputToHidden());
    }
}
